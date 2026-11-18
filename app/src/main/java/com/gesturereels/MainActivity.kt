package com.gesturereels

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast

class MainActivity : Activity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 96, 48, 48)
        }
        layout.addView(TextView(this).apply {
            text = "Flick index finger UP = next reel\nFlick DOWN = previous reel"
            textSize = 18f
        })
        layout.addView(button("1. Enable accessibility service") {
            startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
        })
        layout.addView(button("2. Start gesture control") { start() })
        layout.addView(button("Test swipe (3 sec delay)") {
            if (!SwipeAccessibilityService.isConnected()) {
                Toast.makeText(this, "Accessibility service is OFF", Toast.LENGTH_LONG).show()
            } else {
                Toast.makeText(this, "Open Reels now...", Toast.LENGTH_SHORT).show()
                Handler(Looper.getMainLooper()).postDelayed({ SwipeAccessibilityService.swipe(true) }, 3000)
            }
        })
        layout.addView(button("Stop") {
            stopService(Intent(this, GestureService::class.java))
        })
        setContentView(layout)
    }

    private fun button(label: String, onClick: () -> Unit): View =
        Button(this).apply {
            text = label
            setOnClickListener { onClick() }
        }

    private fun start() {
        val needed = mutableListOf(Manifest.permission.CAMERA)
        if (Build.VERSION.SDK_INT >= 33) needed.add(Manifest.permission.POST_NOTIFICATIONS)
        val missing = needed.filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
        if (missing.isNotEmpty()) {
            requestPermissions(missing.toTypedArray(), 1)
            return
        }
        startForegroundService(Intent(this, GestureService::class.java))
    }

    override fun onRequestPermissionsResult(code: Int, perms: Array<out String>, results: IntArray) {
        super.onRequestPermissionsResult(code, perms, results)
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) start()
    }
}
