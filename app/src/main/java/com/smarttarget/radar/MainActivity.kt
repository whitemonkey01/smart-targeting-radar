package com.smarttarget.radar

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.util.Size
import android.widget.ImageButton
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var statusText: TextView
    private lateinit var debugText: TextView
    private lateinit var objectDetector: ObjectDetectorHelper

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else finish()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        statusText = findViewById(R.id.statusText)
        debugText = findViewById(R.id.debugText)
        val settingsButton = findViewById<ImageButton>(R.id.settingsButton)
        statusText.text = "Status: Starting..."

        objectDetector = ObjectDetectorHelper(
            context = this,
            onDetections = { detections ->
                runOnUiThread { handleDetections(detections) }
            },
            onDebug = { info ->
                runOnUiThread {
                    debugText.text = buildString {
                        append("${info.detectionsCount} det | ${info.inferenceTimeMs}ms")
                        append(" | fmt=${info.imageFormat} p=${info.planesCount}")
                        if (info.errorMessage.isNotEmpty()) {
                            append("\n${info.errorMessage}")
                            debugText.setTextColor(Color.RED)
                        } else {
                            debugText.setTextColor(Color.argb(255, 136, 255, 136))
                        }
                    }
                }
            }
        )

        settingsButton.setOnClickListener { showSettingsDialog() }

        requestCamera()
    }

    private var trackedClassStates = booleanArrayOf()

    private fun showSettingsDialog() {
        if (trackedClassStates.isEmpty()) {
            trackedClassStates = BooleanArray(objectDetector.labels.size) { true }
        }

        val builder = AlertDialog.Builder(this)
        builder.setTitle("Detection Settings")

        val layout = layoutInflater.inflate(R.layout.dialog_settings, null)
        builder.setView(layout)

        val maxDetSeek = layout.findViewById<android.widget.SeekBar>(R.id.maxDetectionsSeek)
        val maxDetLabel = layout.findViewById<TextView>(R.id.maxDetectionsLabel)
        val confSeek = layout.findViewById<android.widget.SeekBar>(R.id.confidenceSeek)
        val confLabel = layout.findViewById<TextView>(R.id.confidenceLabel)
        val classFilterLabel = layout.findViewById<TextView>(R.id.classFilterLabel)
        val selectObjectsBtn = layout.findViewById<android.widget.Button>(R.id.selectObjectsButton)

        maxDetSeek.progress = objectDetector.maxDetections - 1
        maxDetLabel.text = "Max detections: ${objectDetector.maxDetections}"
        maxDetSeek.setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seek: android.widget.SeekBar, value: Int, fromUser: Boolean) {
                val v = value + 1
                maxDetLabel.text = "Max detections: $v"
                objectDetector.maxDetections = v
            }
            override fun onStartTrackingTouch(seek: android.widget.SeekBar) {}
            override fun onStopTrackingTouch(seek: android.widget.SeekBar) {}
        })

        confSeek.progress = (objectDetector.confidenceThreshold * 100).toInt()
        confLabel.text = "Confidence: ${(objectDetector.confidenceThreshold * 100).toInt()}%"
        confSeek.setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seek: android.widget.SeekBar, value: Int, fromUser: Boolean) {
                val v = value.coerceIn(5, 95)
                val f = v / 100f
                confLabel.text = "Confidence: $v%"
                objectDetector.confidenceThreshold = f
            }
            override fun onStartTrackingTouch(seek: android.widget.SeekBar) {}
            override fun onStopTrackingTouch(seek: android.widget.SeekBar) {}
        })

        fun updateClassFilterLabel() {
            val count = trackedClassStates.count { it }
            classFilterLabel.text = "Tracked objects: $count/${trackedClassStates.size}"
        }
        updateClassFilterLabel()

        selectObjectsBtn.setOnClickListener {
            showClassSelectionDialog()
            updateClassFilterLabel()
        }

        builder.setPositiveButton("Done") { dialog, _ ->
            applyClassFilter()
            dialog.dismiss()
        }
        builder.show()
    }

    private fun showClassSelectionDialog() {
        val labels = objectDetector.labels.toTypedArray()
        val states = trackedClassStates.clone()

        AlertDialog.Builder(this)
            .setTitle("Select Objects to Detect")
            .setMultiChoiceItems(labels, states) { _, which, isChecked ->
                states[which] = isChecked
            }
            .setNeutralButton("Select All") { _, _ ->
                states.fill(true)
                trackedClassStates = states
                applyClassFilter()
            }
            .setNegativeButton("None") { _, _ ->
                states.fill(false)
                trackedClassStates = states
                applyClassFilter()
            }
            .setPositiveButton("Apply") { _, _ ->
                trackedClassStates = states
                applyClassFilter()
            }
            .show()
    }

    private fun applyClassFilter() {
        val enabled = mutableSetOf<Int>()
        for (i in trackedClassStates.indices) {
            if (trackedClassStates[i]) enabled.add(i)
        }
        objectDetector.enabledClassIds = enabled
    }

    private fun requestCamera() {
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also { it.surfaceProvider = previewView.surfaceProvider }

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(Executors.newSingleThreadExecutor()) { imageProxy ->
                        objectDetector.detect(imageProxy)
                    }
                }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                imageAnalysis
            )

            statusText.text = "Status: Scanning"
        }, ContextCompat.getMainExecutor(this))
    }

    private fun handleDetections(detections: List<Detection>) {
        overlayView.setDetections(detections)

        if (detections.isEmpty()) {
            statusText.text = "Scanning..."
            statusText.setTextColor(Color.WHITE)
            return
        }

        statusText.text = "Detected: ${detections.joinToString(", ") { it.label }}"
        statusText.setTextColor(Color.GREEN)
    }

    override fun onDestroy() {
        super.onDestroy()
        objectDetector.close()
    }
}
