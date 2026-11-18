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
import android.widget.CheckBox
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast

class MainActivity : Activity() {

    private val prefs by lazy { getSharedPreferences("cfg", MODE_PRIVATE) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 48, 48, 48)
        }
        layout.addView(TextView(this).apply {
            text = "Flick finger UP = next reel\nOpen palm = previous reel\n" +
                "Close eyes 4 sec = show/hide cursor\n" +
                "Move head = move cursor, blink twice = click"
            textSize = 18f
        })
        layout.addView(button("1. Enable accessibility service") {
            startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
        })
        layout.addView(button("2. Allow cursor overlay") {
            if (!Settings.canDrawOverlays(this)) {
                startActivity(Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                    android.net.Uri.parse("package:$packageName")))
            } else {
                Toast.makeText(this, "Already allowed", Toast.LENGTH_SHORT).show()
            }
        })
        layout.addView(button("3. Start gesture control") { start() })
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

        layout.addView(TextView(this).apply {
            text = "Tuning (applies instantly)"
            textSize = 18f
            setPadding(0, 48, 0, 8)
        })
        addSeek(layout, "Cursor speed", "speed", 70, 20) { p -> "%.1f".format(1f + p / 10f) }
        addSeek(layout, "Cursor smoothness", "smooth", 10, 6) { p -> "$p / 10" }
        addSeek(layout, "Blink sensitivity (higher = easier)", "blink", 40, 20) { p -> "$p / 40" }
        layout.addView(CheckBox(this).apply {
            text = "Pause finger/palm gestures while cursor is on"
            isChecked = prefs.getBoolean("pause_hand", false)
            setOnCheckedChangeListener { _, checked ->
                prefs.edit().putBoolean("pause_hand", checked).apply()
            }
        })

        setContentView(ScrollView(this).apply { addView(layout) })
    }

    private fun addSeek(
        parent: LinearLayout,
        title: String,
        key: String,
        maxVal: Int,
        defVal: Int,
        fmt: (Int) -> String
    ) {
        val cur = prefs.getInt(key, defVal)
        val label = TextView(this)
        label.text = title + ": " + fmt(cur)
        label.setPadding(0, 24, 0, 0)
        val bar = SeekBar(this)
        bar.max = maxVal
        bar.progress = cur
        bar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar?, p: Int, fromUser: Boolean) {
                label.text = title + ": " + fmt(p)
                prefs.edit().putInt(key, p).apply()
            }

            override fun onStartTrackingTouch(s: SeekBar?) {}
            override fun onStopTrackingTouch(s: SeekBar?) {}
        })
        parent.addView(label)
        parent.addView(bar)
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
