package com.smarttarget.radar

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

data class Detection(
    val label: String,
    val score: Float,
    val boundingBox: RectF
)

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val labelPaint = Paint().apply {
        color = Color.WHITE
        textSize = 36f
        isAntiAlias = true
        isFakeBoldText = true
    }

    private val backgroundPaint = Paint().apply {
        color = Color.argb(120, 0, 0, 0)
        style = Paint.Style.FILL
    }

    private val detections = mutableListOf<Detection>()

    fun setDetections(results: List<Detection>) {
        detections.clear()
        detections.addAll(results)
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()

        for (detection in detections) {
            val box = detection.boundingBox
            val left = box.left * w
            val top = box.top * h
            val right = box.right * w
            val bottom = box.bottom * h

            val color = getColorForLabel(detection.label)

            val boxPaint = Paint().apply {
                this.color = color
                style = Paint.Style.STROKE
                strokeWidth = 4f
            }
            canvas.drawRect(left, top, right, bottom, boxPaint)

            val labelText = "${detection.label} (%.0f%%)".format(detection.score * 100)
            val textWidth = labelPaint.measureText(labelText)

            val labelTop = (top - 46f).coerceAtLeast(0f)
            backgroundPaint.color = Color.argb(180, Color.red(color), Color.green(color), Color.blue(color))
            canvas.drawRect(
                left,
                labelTop,
                left + textWidth + 16f,
                top,
                backgroundPaint
            )
            labelPaint.color = Color.WHITE
            canvas.drawText(labelText, left + 8f, top - 10f, labelPaint)
        }
    }

    private fun getColorForLabel(label: String): Int {
        val idx = label.hashCode() and 0x7FFFFFFF
        return COLORS[idx % COLORS.size]
    }

    companion object {
        private val COLORS = intArrayOf(
            Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW, Color.CYAN,
            Color.MAGENTA, Color.rgb(255, 165, 0), Color.rgb(128, 0, 128),
            Color.rgb(0, 255, 127), Color.rgb(255, 20, 147),
            Color.rgb(0, 191, 255), Color.rgb(255, 215, 0),
            Color.rgb(50, 205, 50), Color.rgb(255, 69, 0),
            Color.rgb(147, 112, 219), Color.rgb(0, 255, 255)
        )
    }
}
