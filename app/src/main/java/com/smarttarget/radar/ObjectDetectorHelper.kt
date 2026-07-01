package com.smarttarget.radar

import android.content.Context
import android.graphics.RectF
import android.media.Image
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

private const val LABELS_FILE = "coco_labels.txt"
private const val MODEL_FILE = "yolov8n.tflite"
private const val MODEL_INPUT_SIZE = 640

data class DebugInfo(
    val inferenceTimeMs: Long,
    val detectionsCount: Int,
    val errorMessage: String = "",
    val imageFormat: Int = 0,
    val planesCount: Int = 0
)

class ObjectDetectorHelper(
    private val context: Context,
    private val onDetections: (List<Detection>) -> Unit,
    private val onDebug: ((DebugInfo) -> Unit)? = null
) {
    private var interpreter: Interpreter? = null
    var labels: List<String> = emptyList()
        private set
    private var modelInputSize = MODEL_INPUT_SIZE
    private var initError: String? = null

    var confidenceThreshold: Float = 0.25f
    var iouThreshold: Float = 0.5f
    var maxDetections: Int = 5
    var enabledClassIds: Set<Int>? = null

    init {
        try {
            val modelBuffer = loadModelFile()
            interpreter = Interpreter(modelBuffer)

            val inputShape = interpreter?.getInputTensor(0)?.shape()
            Log.d("YOLO", "Model input shape: ${inputShape?.contentToString()}")

            modelInputSize = if (inputShape != null && inputShape.size >= 2) {
                inputShape[1]
            } else MODEL_INPUT_SIZE

            val inputType = interpreter?.getInputTensor(0)?.dataType()
            Log.d("YOLO", "Input type: $inputType, size: ${modelInputSize}x$modelInputSize")

            val outputShape = interpreter?.getOutputTensor(0)?.shape()
            Log.d("YOLO", "Model output shape: ${outputShape?.contentToString()}")

            labels = loadLabels()
            Log.d("YOLO", "Loaded ${labels.size} labels")
        } catch (e: Exception) {
            initError = "Init: ${e.message ?: e.javaClass.simpleName}"
            Log.e("YOLO", "Init failed", e)
        }
    }

    private fun loadModelFile(): MappedByteBuffer {
        val afd = context.assets.openFd(MODEL_FILE)
        val inputStream = FileInputStream(afd.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = afd.startOffset
        val declaredLength = afd.declaredLength
        val buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        inputStream.close()
        return buffer
    }

    private fun loadLabels(): List<String> {
        return try {
            context.assets.open(LABELS_FILE).bufferedReader().readLines()
        } catch (e: Exception) {
            Log.w("YOLO", "Labels file not found")
            emptyList()
        }
    }

    fun detect(imageProxy: androidx.camera.core.ImageProxy) {
        val image = imageProxy.image
        if (image == null) {
            imageProxy.close()
            onDebug?.invoke(DebugInfo(0, 0, "No camera frame"))
            onDetections(emptyList())
            return
        }

        val interpreter = interpreter
        if (interpreter == null) {
            imageProxy.close()
            val msg = initError ?: "Model not loaded"
            onDebug?.invoke(DebugInfo(0, 0, msg))
            onDetections(emptyList())
            return
        }

        try {
            val startTime = System.currentTimeMillis()
            val fmt = image.format
            val nPlanes = image.planes.size

            val inputBuffer = ByteBuffer.allocateDirect(4 * modelInputSize * modelInputSize * 3)
            inputBuffer.order(ByteOrder.nativeOrder())

            imageToFloatBuffer(image, inputBuffer)

            val outputShape = interpreter.getOutputTensor(0).shape()

            if (outputShape.size == 3 && outputShape[1] == 84) {
                val output = Array(1) { Array(84) { FloatArray(outputShape[2]) } }
                interpreter.run(inputBuffer, output)
                val detections = parseYOLOv8Output(output[0], image.width, image.height)
                val elapsed = System.currentTimeMillis() - startTime
                Log.d("YOLO", "Inference: ${elapsed}ms, detections: ${detections.size}")
                val debugMsg = if (detections.isEmpty()) "No objects >${(confidenceThreshold*100).toInt()}% conf" else ""
                onDebug?.invoke(DebugInfo(elapsed, detections.size, debugMsg, fmt, nPlanes))
                imageProxy.close()
                onDetections(detections)
            } else if (outputShape.size == 3 && outputShape[2] == 84) {
                val output = Array(1) { Array(outputShape[1]) { FloatArray(84) } }
                interpreter.run(inputBuffer, output)
                val transposed = Array(84) { i -> FloatArray(outputShape[1]) { j -> output[0][j][i] } }
                val detections = parseYOLOv8Output(transposed, image.width, image.height)
                val elapsed = System.currentTimeMillis() - startTime
                Log.d("YOLO", "Inference: ${elapsed}ms, detections: ${detections.size}")
                val debugMsg = if (detections.isEmpty()) "No objects >${(confidenceThreshold*100).toInt()}% conf" else ""
                onDebug?.invoke(DebugInfo(elapsed, detections.size, debugMsg, fmt, nPlanes))
                imageProxy.close()
                onDetections(detections)
            } else {
                val msg = "Unexpected output shape: ${outputShape.contentToString()}"
                Log.e("YOLO", msg)
                onDebug?.invoke(DebugInfo(System.currentTimeMillis() - startTime, 0, msg, fmt, nPlanes))
                imageProxy.close()
                onDetections(emptyList())
            }
        } catch (e: Exception) {
            val msg = "Error: ${e.message ?: e.javaClass.simpleName}"
            Log.e("YOLO", "Detection error", e)
            onDebug?.invoke(DebugInfo(0, 0, msg, image.format, image.planes.size))
            imageProxy.close()
            onDetections(emptyList())
        }
    }

    private fun imageToFloatBuffer(image: Image, outBuf: ByteBuffer) {
        val w = image.width
        val h = image.height
        val planes = image.planes
        val nPlanes = planes.size

        outBuf.rewind()

        if (nPlanes == 1) {
            rgbaToFloatBuffer(planes[0].buffer, planes[0].rowStride, planes[0].pixelStride, w, h, outBuf)
        } else {
            yuvToFloatBuffer(planes, w, h, outBuf)
        }
        outBuf.rewind()
    }

    private fun rgbaToFloatBuffer(buf: ByteBuffer, rowStride: Int, pixStride: Int, imgW: Int, imgH: Int, outBuf: ByteBuffer) {
        val cap = buf.capacity()
        for (row in 0 until modelInputSize) {
            for (col in 0 until modelInputSize) {
                val srcRow = row * imgH / modelInputSize
                val srcCol = col * imgW / modelInputSize
                val idx = srcRow * rowStride + srcCol * pixStride

                val r = if (idx + 2 < cap) (buf.get(idx).toInt() and 0xFF) / 255f else 0f
                val g = if (idx + 2 < cap) (buf.get(idx + 1).toInt() and 0xFF) / 255f else 0f
                val b = if (idx + 2 < cap) (buf.get(idx + 2).toInt() and 0xFF) / 255f else 0f

                outBuf.putFloat(r); outBuf.putFloat(g); outBuf.putFloat(b)
            }
        }
    }

    private fun yuvToFloatBuffer(planes: Array<Image.Plane>, imgW: Int, imgH: Int, outBuf: ByteBuffer) {
        val yBuf = planes[0].buffer
        val uBuf = planes[1].buffer
        val vBuf = if (planes.size >= 3) planes[2].buffer else planes[1].buffer
        val yRowStride = planes[0].rowStride
        val uRowStride = planes[1].rowStride
        val vRowStride = if (planes.size >= 3) planes[2].rowStride else planes[1].rowStride
        val uPixStride = planes[1].pixelStride
        val vPixStride = if (planes.size >= 3) planes[2].pixelStride else planes[1].pixelStride

        val yCap = yBuf.capacity()
        val uCap = uBuf.capacity()
        val vCap = vBuf.capacity()

        val isSemiPlanar = uPixStride == 2 || vPixStride == 2

        for (row in 0 until modelInputSize) {
            for (col in 0 until modelInputSize) {
                val srcRow = row * imgH / modelInputSize
                val srcCol = col * imgW / modelInputSize
                val yIdx = srcRow * yRowStride + srcCol

                val y = if (yIdx < yCap) (yBuf.get(yIdx).toInt() and 0xFF) else 128

                val uvRow = srcRow / 2
                val uvCol = srcCol / 2
                val u: Int
                val v: Int

                if (isSemiPlanar) {
                    val chromaBuf = if (uPixStride == 2) uBuf else vBuf
                    val chromaRowStride = if (uPixStride == 2) uRowStride else vRowStride
                    val chromaCap = if (uPixStride == 2) uCap else vCap
                    val chromaIdx = uvRow * chromaRowStride + uvCol * 2
                    v = if (chromaIdx < chromaCap) (chromaBuf.get(chromaIdx).toInt() and 0xFF) else 128
                    u = if (chromaIdx + 1 < chromaCap) (chromaBuf.get(chromaIdx + 1).toInt() and 0xFF) else 128
                } else {
                    val uIdx = uvRow * uRowStride + uvCol * uPixStride
                    val vIdx = uvRow * vRowStride + uvCol * vPixStride
                    u = if (uIdx < uCap) (uBuf.get(uIdx).toInt() and 0xFF) else 128
                    v = if (vIdx < vCap) (vBuf.get(vIdx).toInt() and 0xFF) else 128
                }

                val r = (y + 1.402f * (v - 128)).coerceIn(0f, 255f) / 255f
                val g = (y - 0.344f * (u - 128) - 0.714f * (v - 128)).coerceIn(0f, 255f) / 255f
                val b = (y + 1.773f * (u - 128)).coerceIn(0f, 255f) / 255f

                outBuf.putFloat(r); outBuf.putFloat(g); outBuf.putFloat(b)
            }
        }
    }

    private fun parseYOLOv8Output(output: Array<FloatArray>, imgWidth: Int, imgHeight: Int): List<Detection> {
        val scores = mutableListOf<Float>()
        val boxes = mutableListOf<FloatArray>()
        val classIds = mutableListOf<Int>()

        val numDetections = output[0].size
        val scaleX = imgWidth.toFloat() / modelInputSize
        val scaleY = imgHeight.toFloat() / modelInputSize

        for (i in 0 until numDetections) {
            var maxScore = 0f
            var maxClassId = -1
            for (c in 4 until 84) {
                val score = sigmoid(output[c][i])
                if (score > maxScore) {
                    maxScore = score
                    maxClassId = c - 4
                }
            }

            if (maxScore < confidenceThreshold) continue
            val enabled = enabledClassIds
            if (enabled != null && maxClassId !in enabled) continue

            val cx = output[0][i]
            val cy = output[1][i]
            val w = output[2][i]
            val h = output[3][i]

            val left = (cx - w / 2) * scaleX
            val top = (cy - h / 2) * scaleY
            val right = (cx + w / 2) * scaleX
            val bottom = (cy + h / 2) * scaleY

            scores.add(maxScore)
            boxes.add(floatArrayOf(left, top, right, bottom))
            classIds.add(maxClassId)
        }

        val indices = nms(boxes, scores)

        return indices.map { idx ->
            val box = boxes[idx]
            val label = if (classIds[idx] < labels.size) labels[classIds[idx]] else "Class${classIds[idx]}"
            Detection(
                label = label,
                score = scores[idx],
                boundingBox = RectF(box[0] / imgWidth, box[1] / imgHeight, box[2] / imgWidth, box[3] / imgHeight)
            )
        }
    }

    private fun sigmoid(x: Float): Float = 1.0f / (1.0f + kotlin.math.exp(-x))

    private fun nms(boxes: List<FloatArray>, scores: List<Float>): List<Int> {
        val indices = scores.indices.sortedByDescending { scores[it] }
        val selected = mutableListOf<Int>()
        for (i in indices) {
            var keep = true
            for (j in selected) {
                if (iou(boxes[i], boxes[j]) > iouThreshold) {
                    keep = false
                    break
                }
            }
            if (keep) {
                selected.add(i)
                if (selected.size >= maxDetections) break
            }
        }
        return selected
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val interLeft = maxOf(a[0], b[0])
        val interTop = maxOf(a[1], b[1])
        val interRight = minOf(a[2], b[2])
        val interBottom = minOf(a[3], b[3])
        if (interRight <= interLeft || interBottom <= interTop) return 0f
        val interArea = (interRight - interLeft) * (interBottom - interTop)
        val areaA = (a[2] - a[0]) * (a[3] - a[1])
        val areaB = (b[2] - b[0]) * (b[3] - b[1])
        return interArea / (areaA + areaB - interArea)
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
