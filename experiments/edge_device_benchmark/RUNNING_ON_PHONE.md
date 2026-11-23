# Running the BioDINO-Edge benchmark on your phone

This guide collects the real-device edge measurements requested by the reviewer
(end-to-end latency, peak RAM, model load time, energy, and the adaptation-cost
breakdown) from a physical Android phone.

Everything is already built. You only need a phone, a USB cable, and to run one script.

---

## 1. What you need

| Item | Status / requirement |
|---|---|
| Android phone | 64-bit ARM (`arm64-v8a`), Android 7.0 (API 24) or newer. Most phones from 2019+ qualify. |
| USB cable | Data-capable (not charge-only). |
| Benchmark APK | Built: `android_app\app\build\outputs\apk\benchmark\BioDINO-Edge-v1.0.1-benchmark.apk` |
| Tier-2 companion model | Present: `biodino_companion_int8.onnx` (85.3 MB, repo root) |
| adb | Present in your Android SDK; the script finds it automatically. |
| Python | The repo `.venv` is used automatically for table generation. |

The benchmark runs **CPU-only, INT8, 4 threads, NNAPI off**, matching the paper's testbed.

---

## 2. Enable USB debugging on the phone

1. Open **Settings → About phone**.
2. Tap **Build number** 7 times (you'll see "You are now a developer").
3. Go back to **Settings → System → Developer options**.
4. Turn on **USB debugging**.
5. (Recommended) Turn on **Stay awake** so the screen doesn't sleep while charging during the run.

---

## 3. Connect and authorize

1. Plug the phone into the PC with the USB cable.
2. On the phone, a dialog **"Allow USB debugging?"** appears — tick *Always allow* and tap **Allow**.
3. Open PowerShell in the repo root and verify the device is seen:

   ```powershell
   cd d:\Research\uncertainty-gated-paper
   & "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" devices
   ```

   You should see one line ending in `device`, e.g.:

   ```
   List of devices attached
   R58M123ABCD    device
   ```

   - If it says `unauthorized`: unlock the phone and accept the Allow dialog again.
   - If the list is empty: try a different cable/USB port, or re-toggle USB debugging.

---

## 4. Remove any existing BioDINO app (important)

The benchmark APK is debug-signed. If you already installed the **release** app
(different signature), installing over it fails. Uninstall first (safe — this only
removes the app, not your phone data):

```powershell
& "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" uninstall com.biodino.edge
```

Ignore the error if it says the package is not installed.

---

## 5. Run the benchmark

From the repo root:

```powershell
.\experiments\edge_device_benchmark\run_device_benchmark.ps1 -SustainedSeconds 600
```

The script automatically:
1. Installs the benchmark APK.
2. Pushes the 85.3 MB companion model to the phone.
3. Resets battery statistics (for the energy estimate).
4. Launches the benchmark screen on the phone.
5. Waits for it to finish, then pulls the results and generates LaTeX tables.

**How long:** roughly 12–15 minutes. The bulk is the 600 s (10 min) sustained loop
used for energy/thermal, plus ~30 s to extract 440 support views for the adaptation
breakdown. Keep the phone plugged in and the screen on; don't lock it.

**On the phone** you'll see a black-on-white log screen ("BioDINO-Edge on-device
benchmark") printing progress: model load, latency loop, extraction, sustained loop,
then `BENCHMARK_DONE`.

> If the script can't find adb, pass it explicitly:
> ```powershell
> .\experiments\edge_device_benchmark\run_device_benchmark.ps1 -Adb "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" -SustainedSeconds 600
> ```

### Quick test (no long energy loop)

To verify everything works in ~2 minutes before the full run:

```powershell
.\experiments\edge_device_benchmark\run_device_benchmark.ps1 -SustainedSeconds 120 -SkipEnergy
```

---

## 6. Outputs

Written to `experiments\edge_device_benchmark\results\`:

| File | Contents |
|---|---|
| `benchmark_report.json` | All raw measurements (device spec, load time, RAM, per-stage latency, adaptation breakdown, sustained loop). |
| `batterystats.txt` | Battery dump used for the energy estimate. |
| `edge_device_tables.tex` | Three booktabs tables ready for the paper: `tab:device_spec`, `tab:device_perf`, `tab:adaptation_breakdown`. |

**Send me `benchmark_report.json`** (or the `.tex`) and I will insert the measured
numbers into the paper, rescope the "<22 ms" claim, and write the reviewer rebuttal.

---

## 7. What each measurement answers

- **End-to-end latency** — `latency.cascade_end_to_end_ms` and `tier2_on_all_end_to_end_ms` (mean±std / median / p95).
- **Peak RAM** — `memory.peak_pss_mb` (proportional set size, the standard Android memory metric).
- **Model load time** — `model_load.tier1_total_ms`, `tier2_total_ms`.
- **Energy** — mJ/inference in the performance table, from the sustained loop + batterystats (best-effort; shown as `--` if it can't be attributed to the app).
- **"<22 ms" breakdown** — `adaptation`: feature extraction (440 views) is reported **separately** from covariance → Ledoit–Wolf shrinkage → Cholesky → projection solve. `algebra_total_ms` is the recalibration budget the claim refers to.

---

## 8. Manual commands (fallback / understanding)

If you prefer to run the steps yourself instead of the script:

```powershell
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
$pkg = "com.biodino.edge"
$remote = "/sdcard/Android/data/$pkg/files"

# Install + push companion
& $adb install -r "android_app\app\build\outputs\apk\benchmark\BioDINO-Edge-v1.0.1-benchmark.apk"
& $adb shell mkdir -p $remote
& $adb push "biodino_companion_int8.onnx" "$remote/biodino_companion_int8.onnx"

# Energy baseline
& $adb shell dumpsys batterystats --reset

# Launch (100 latency iters, 600 s sustained loop)
& $adb logcat -c
& $adb shell am start -n "$pkg/.benchmark.BenchmarkActivity" --ei n_iters 100 --ei sustained_seconds 600

# Watch progress (Ctrl+C to stop watching; the benchmark keeps running)
& $adb logcat -s BioDinoBench:*

# After you see BENCHMARK_DONE, pull results
& $adb pull "$remote/benchmark_report.json" "experiments\edge_device_benchmark\results\benchmark_report.json"
& $adb shell dumpsys batterystats > "experiments\edge_device_benchmark\results\batterystats.txt"

# Generate tables
.venv\Scripts\python.exe experiments\edge_device_benchmark\make_tables.py --report experiments\edge_device_benchmark\results\benchmark_report.json --batterystats experiments\edge_device_benchmark\results\batterystats.txt --out experiments\edge_device_benchmark\results\edge_device_tables.tex
```

### Tunable intent extras

| Extra | Default | Meaning |
|---|---|---|
| `n_iters` | 100 | Latency samples (after 10 warm-up). |
| `sustained_seconds` | 120 | Sustained-loop duration for energy/thermal. Use 600 for a stable energy estimate. |
| `tau` | 0.35 | Gate threshold. |
| `adaptation_views` | 440 | Support views for the adaptation breakdown. |
| `adaptation_repeats` | 5 | Algebra repeat count for mean±std. |
| `cooldown_seconds` | 60 | Idle cool-down before timing the adaptation algebra (keeps it out of thermal throttling). |
| `run_adaptation` | 1 | Set 0 to skip the adaptation benchmark. |

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `No adb device found` | Reconnect cable; accept the "Allow USB debugging" dialog; `adb devices` must show `device`. |
| `INSTALL_FAILED_UPDATE_INCOMPATIBLE` | Signature clash with an existing app. Run `adb uninstall com.biodino.edge`, then re-run. |
| `INSTALL_FAILED_NO_MATCHING_ABIS` | The phone is 32-bit. The APK targets `arm64-v8a`; use a 64-bit phone. |
| `adb` not recognized | Use the full path (Section 5 note) or the script auto-detects it. |
| Log shows "Tier-2 companion not found" | The push failed. Re-run; confirm `biodino_companion_int8.onnx` exists at the repo root. Without it, Tier-2 and adaptation stages are skipped. |
| Adaptation skipped / `Cholesky failed` | Send me `benchmark_report.json` and the logcat; the shrunk covariance should be positive definite, so this indicates a data issue to inspect. |
| Energy shows `--` | batterystats couldn't attribute power to the app (device-dependent). Latency/RAM/load are still valid. A USB power meter gives cleaner energy if needed. |
| Phone gets hot / throttles | Expected during the sustained loop; `thermal_max` records it. Let the phone cool between runs. |
| Script times out | The sustained loop is long; increase patience or re-run with a smaller `-SustainedSeconds`. |

To capture a full log for debugging:

```powershell
& "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" logcat -d -s BioDinoBench:* > experiments\edge_device_benchmark\results\logcat.txt
```

---

## 10. After the run

Send me `experiments\edge_device_benchmark\results\benchmark_report.json`. I will:
1. Check the measured `T1`/`T2` against the paper's current constants (21.12 / 64.02 ms) and, if they differ materially, update them and re-derive all dependent latencies consistently.
2. Insert the three measured tables and rewrite the Edge Hardware Testbed section for the real device.
3. Rescope the "<22 ms" claim to the algebraic recalibration budget (extraction reported separately).
4. Write the point-by-point reviewer rebuttal.