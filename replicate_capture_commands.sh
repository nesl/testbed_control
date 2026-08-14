#!/usr/bin/env bash

# Command cookbook for replicate_capture.py.
#
# The script captures each labels.csv sample at one fixed 5600K Amaran light
# position using brightness values 10, 200, 500, 700, and 1000.  At every
# brightness it records three auto-exposure images followed by the paper's 27
# manual exposure combinations. After the full sweep at z001, it pauses so you
# can manually change zoom, then repeats the full sweep at z002.
#
# This file is intentionally safe to run: it only prints instructions. Copy a
# commented command below and run it from the repository root.

echo "This is a command cookbook. Copy a commented command and run it manually."
echo "No camera or light command was executed."
exit 0


# Show the default full 15-sample / 2-zoom / 4,500-image plan without files.
# python replicate_capture.py --plan-only

# Print the complete plan manifest, including p001-p027 mappings.
# python replicate_capture.py --plan-only --print-plan-json

# Simulate one sample, two zooms, and all five brightness levels.
# python replicate_capture.py --dry-run --yes --sample-id ILSVRC2012_val_00038504 --output-dir /private/tmp/replicate-capture-dry-run

# Real collection. Like control.py, the testbed's default Amaran key is already
# configured. The script prompts before each zoom position; after z001
# finishes, manually change zoom and press Enter for z002. Only after both
# complete does it prompt for the next printed sample.
# Set focus mode and Multi Metering on the camera/lens before starting; the
# script preserves those settings. It uses P mode for AE and M for p001-p027.
# sonycam replicate_capture.py --output-dir data_replicate/replicated_capture

# Use three manual zoom positions per sample instead of the default two.
# sonycam replicate_capture.py --zoom-count 3 --output-dir data_replicate/replicated_capture_three_zooms

# Resume after any interruption by running exactly the same command again.
# Valid existing JPEGs are skipped and only missing/corrupt captures are taken.
# sonycam replicate_capture.py --output-dir data_replicate/replicated_capture

# Capture or resume only one selected printed sample in the same full dataset.
# sonycam replicate_capture.py --sample-id ILSVRC2012_val_00038504 --output-dir data_replicate/replicated_capture

# Optional: save to both the host and the camera card.
# sonycam replicate_capture.py --save-media host-and-camera --output-dir data_replicate/replicated_capture
