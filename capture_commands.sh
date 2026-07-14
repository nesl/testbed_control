#!/usr/bin/env bash

# Command cookbook for control.py.
# This file is intentionally just a commented command list. Copy a command,
# edit class_id / output-dir / position / params, then run it from the repo root.
#
# New collection pipeline:
#   1. Physically set the light position.
#   2. Run control.py with --light-position POSITION.
#   3. Enter class IDs interactively; after each input, capture starts after
#      --start-delay-seconds seconds.
#   4. Stop the script, move the light, rerun with a new --light-position.
#
# Mapping layout:
#   <output-dir>/maps/parameters.json  cumulative global parameter map
#     samples[] includes per-object timing: started_at, completed_at,
#     object_elapsed_seconds, capture_elapsed_seconds
#     params[] includes p000 for auto exposure when enabled
#   <output-dir>/maps/captures.jsonl   cumulative per-image records

echo "This is a command cookbook. Copy a commented command from this file and run it manually."
echo "No capture command was executed."
exit 0


# =========================
# Command During Test
# =========================

# Reflect light, single view, iterative class input, intensity 10, p000 auto exposure.
# Prints per-object timing at the end of each sample.
# sonycam control.py --capture-mode append --output-dir dataset --light-position reflect --light-intensities 10 --fast-shutter --skip-turntable --views 1

# Face light can use a different intensity from normal/reflect light.
# Append another physical light position to an existing dataset. Prompts for class_id repeatedly.
# sonycam control.py --capture-mode append --output-dir dataset --light-position side --fast-shutter

# Append one class only for the current physical light position.
# sonycam control.py --capture-mode append --class-id 1 --output-dir dataset --light-position front --fast-shutter

# Camera + light only for the current light position, skipping turntable.
# sonycam control.py --capture-mode append --class-id 1 --output-dir dataset --light-position front --fast-shutter --skip-turntable


# delete records
# check only 
# python capture_util/remove_records.py --output-dir dataset --class-id 1 --sample-id 3

#apply
# sudo 

# =========================
# Help / Sanity Checks
# =========================

# Show all control.py arguments.
# python control.py --help

# Same help command through the Sony camera launcher.
# sonycam control.py --help

# Check Python syntax after editing control.py.
# python -m py_compile control.py

# Pretty-print the current global parameter map.
# python -m json.tool dataset/maps/parameters.json

# Print the first few per-image JSONL records.
# python -c 'from itertools import islice; from pathlib import Path; print("".join(islice(Path("dataset/maps/captures.jsonl").open(), 5)), end="")'


# =========================
# Dataset Cleanup
# =========================

# Preview removing one bad sample and its map records.
# python capture_util/remove_records.py --output-dir dataset --class-id 1 --sample-id 2

# Actually remove one bad sample and its map records.
# python capture_util/remove_records.py --output-dir dataset --class-id 1 --sample-id 2 --apply

# Remove an entire class.
# python capture_util/remove_records.py --output-dir dataset --class-id 2 --apply

# Remove one light id across the dataset.
# python capture_util/remove_records.py --output-dir dataset --light-id 1 --apply

# Remove one physical lighting condition across the dataset.
# python capture_util/remove_records.py --output-dir dataset --light-position reflect --intensity 10 --cct 5600 --apply

# Remove one view id across the dataset.
# python capture_util/remove_records.py --output-dir dataset --view-id 1 --apply

# Remove only p000 auto exposure images/records.
# python capture_util/remove_records.py --output-dir dataset --param-id 0 --apply


# =========================
# Fresh Dataset
# =========================

# Fresh dry-run into a new empty temp-like directory.
# python control.py --dry-run --capture-mode fresh --class-id 1 --output-dir /private/tmp/testbed-fresh-dry-run --light-position front --light-intensities 10,20 --views 1 --apertures 2.8 --isos 100 --shutters 60 --start-delay-seconds 0 --skip-turntable

# Fresh real capture, one quick image per intensity for one class.
# sonycam control.py --capture-mode fresh --class-id 1 --output-dir dataset_fresh_quick --light-position front --light-intensities 10,20 --views 1 --apertures 2.8 --isos 100 --shutters 60 --start-delay-seconds 0 --fast-shutter


# =========================
# Append Dataset
# =========================

# Append dry-run: same position, two intensities.
# python control.py --dry-run --capture-mode append --class-id 1 --output-dir dataset --light-position front --light-intensities 10,20 --views 1 --apertures 2.8 --isos 100 --shutters 60 --start-delay-seconds 0 --skip-turntable

# Append using the default full intensity sweep for one position.
# sonycam control.py --capture-mode append --class-id 1 --output-dir dataset --light-position front --fast-shutter

# Append a new physical light position. The map will assign different light_id values.
# sonycam control.py --capture-mode append --class-id 1 --output-dir dataset --light-position side --fast-shutter


# =========================
# Smaller Capture Plans
# =========================

# One quick real capture per light intensity before a long run.
# sonycam control.py --capture-mode append --class-id 1 --output-dir dataset --light-position front --views 1 --apertures 2.8 --isos 100 --shutters 60 --start-delay-seconds 0 --fast-shutter

# One view only, with the full camera parameter sweep.
# sonycam control.py --capture-mode append --class-id 1 --output-dir dataset --light-position front --views 1 --fast-shutter

# One view only, with no p000 auto-exposure image.
# sonycam control.py --capture-mode append --class-id 1 --output-dir dataset --light-position front --views 1 --skip-auto-exposure --fast-shutter

# Four views, one camera parameter.
# sonycam control.py --capture-mode append --class-id 1 --output-dir dataset --light-position front --views 4 --view-step 90 --apertures 2.8 --isos 100 --shutters 60 --fast-shutter

# Custom camera parameter sweep.
# sonycam control.py --capture-mode append --class-id 1 --output-dir dataset --light-position front --apertures 2.8,4,8 --isos 100,800,3200 --shutters 60,250 --fast-shutter

# Custom light intensities for current position.
# sonycam control.py --capture-mode append --class-id 1 --output-dir dataset --light-position front --light-intensities 100,300,700 --fast-shutter

# Host + camera storage: save images to both computer and camera card.
# sonycam control.py --capture-mode append --class-id 1 --output-dir dataset --light-position front --save-media host-and-camera --fast-shutter


# =========================
# Per-object Timing Checks
# =========================

# Minimal dry-run timing check: writes samples[].object_elapsed_seconds in parameters.json.
# python control.py --dry-run --capture-mode fresh --class-id 1 --output-dir /private/tmp/testbed-object-timing-dry-run --light-position front --light-intensities 100 --views 1 --apertures 2.8 --isos 100 --shutters 60 --start-delay-seconds 0 --skip-turntable

# Minimal real hardware timing check with p000 auto exposure and one manual parameter.
# sonycam control.py --capture-mode append --class-id 1 --output-dir /private/tmp/testbed-object-timing-hardware --light-position front --light-intensities 100 --views 1 --apertures 2.8 --isos 100 --shutters 60 --start-delay-seconds 0 --settle-seconds 0.2 --fast-shutter

# Real light + camera timing, skip turntable.
# sonycam control.py --capture-mode append --class-id 1 --output-dir /private/tmp/testbed-object-timing-light-camera --light-position front --light-intensities 0,100,500 --views 1 --apertures 2.8 --isos 100 --shutters 60 --start-delay-seconds 0 --settle-seconds 0.2 --skip-turntable --fast-shutter

# Real camera timing only, skip light and turntable.
# sonycam control.py --capture-mode append --class-id 1 --output-dir /private/tmp/testbed-object-timing-camera-only --light-position front --light-intensities 0 --views 1 --apertures 2.8 --isos 100,800 --shutters 60 --start-delay-seconds 0 --skip-light --skip-turntable --fast-shutter


# =========================
# Hardware / Connection Options
# =========================

# Explicit turntable serial port.
# sonycam control.py --capture-mode append --class-id 1 --output-dir dataset --light-position front --turntable-port /dev/cu.usbmodem1101 --fast-shutter

# Custom turntable speed and settle time.
# sonycam control.py --capture-mode append --class-id 1 --output-dir dataset --light-position front --turntable-speed 15 --settle-seconds 1.0 --fast-shutter

# Custom Amaran websocket URL.
# sonycam control.py --capture-mode append --class-id 1 --output-dir dataset --light-position front --light-ws-url ws://127.0.0.1:12345 --fast-shutter

# Custom Amaran client id.
# sonycam control.py --capture-mode append --class-id 1 --output-dir dataset --light-position front --light-client-id 1 --fast-shutter

# Custom Amaran API secret key.
# sonycam control.py --capture-mode append --class-id 1 --output-dir dataset --light-position front --light-api-secret-key YOUR_SECRET_KEY --fast-shutter

# Increase capture timeout if camera transfer is slow.
# sonycam control.py --capture-mode append --class-id 1 --output-dir dataset --light-position front --capture-timeout 60 --fast-shutter

# Remove the initial countdown delay.
# sonycam control.py --capture-mode append --class-id 1 --output-dir dataset --light-position front --start-delay-seconds 0 --fast-shutter
