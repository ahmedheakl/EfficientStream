#!/usr/bin/env bash
# Copies the exported models (ios/models, from export_ios.py) into the EfficientStream app's Documents/models
# on a connected iPhone. Install and open the app once first (Xcode Run), then:
#
#   ios/tools/install_models.sh                 # first connected iPhone
#   ios/tools/install_models.sh "My iPhone"     # by name, UDID, or serial
#
# Unchanged files are skipped, so re-running after an app update is quick.
set -euo pipefail
IOS="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="${MODELS:-$IOS/models}"
BUNDLE_ID="${BUNDLE_ID:-$(sed -n 's/^PRODUCT_BUNDLE_IDENTIFIER *= *//p' "$IOS/app/Configuration/Build.xcconfig" | tr -d '[:space:]')}"

for f in minicpm/model.safetensors mobileclip/ImageEncoder.mlpackage efficientstream_config.json; do
  [ -e "$MODELS/$f" ] || { echo "missing $MODELS/$f: run ios/tools/export_ios.py first" >&2; exit 1; }
done

DEVICE="${1:-}"
if [ -z "$DEVICE" ]; then
  DEVICE="$(xcrun devicectl list devices --json-output /dev/stdout 2>/dev/null | python3 -c '
import json, sys
for d in json.load(sys.stdin)["result"]["devices"]:
    if d.get("connectionProperties", {}).get("tunnelState") == "connected" and d.get("hardwareProperties", {}).get("platform") == "iOS":
        print(d["hardwareProperties"]["udid"]); break')"
fi
[ -n "$DEVICE" ] || { echo "no connected iPhone found (unlock it and check Xcode > Window > Devices)" >&2; exit 1; }

echo "Copying $(du -sh "$MODELS" | cut -f1) of models to $BUNDLE_ID on $DEVICE ..."
xcrun devicectl device copy to --device "$DEVICE" --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" \
  --source "$MODELS" --destination Documents/models
echo "Done. In the app, tap \"Check again\" (or relaunch it)."
