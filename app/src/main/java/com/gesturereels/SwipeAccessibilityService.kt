package com.gesturereels

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.content.Intent
import android.graphics.Path
import android.view.accessibility.AccessibilityEvent

class SwipeAccessibilityService : AccessibilityService() {

    companion object {
        @Volatile
        private var instance: SwipeAccessibilityService? = null

        fun isConnected() = instance != null

        fun swipe(next: Boolean) {
            instance?.doSwipe(next)
        }

        fun goBack() {
            instance?.performGlobalAction(GLOBAL_ACTION_BACK)
        }
    }

    override fun onServiceConnected() {
        instance = this
    }

    override fun onUnbind(intent: Intent?): Boolean {
        instance = null
        return super.onUnbind(intent)
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {}
    override fun onInterrupt() {}

    private fun doSwipe(next: Boolean) {
        val dm = resources.displayMetrics
        val x = dm.widthPixels / 2f
        val top = dm.heightPixels * 0.3f
        val bottom = dm.heightPixels * 0.7f
        val path = Path()
        if (next) {
            path.moveTo(x, bottom); path.lineTo(x, top)
        } else {
            path.moveTo(x, top); path.lineTo(x, bottom)
        }
        val gesture = GestureDescription.Builder()
            .addStroke(GestureDescription.StrokeDescription(path, 0, 200))
            .build()
        dispatchGesture(gesture, null, null)
    }
}
