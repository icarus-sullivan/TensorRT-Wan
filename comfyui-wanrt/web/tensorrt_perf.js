// Front-end companion for nodes/tensorrt_perf.py's TensorRTDiffusionLoader/TensorRTCheckpointLoader.
//
// Purely cosmetic, purely additive: the Python side already ignores magcache_thresh/magcache_K/
// magcache_retention_ratio unless MagCache == "Custom" (see _magcache_params in tensorrt_perf.py).
// Without this file those three widgets just sit there showing stale generic defaults, which reads
// as "you need to configure these" even when picking a preset like Quality. This file makes them
// track the selected preset's real values instead, so nothing needs to be touched for
// Fast/Balanced/Quality. If this file fails to load or errors, the node still behaves correctly --
// it just won't auto-sync those three display values.
//
// Only loads if this whole comfyui-wanrt/ package (not just tensorrt_perf.py dragged in alone) is
// installed, since ComfyUI only serves a WEB_DIRECTORY declared by a package's top-level
// __init__.py -- see that file's docstring for the single-file-drag-and-drop tradeoff this implies.

import { app } from "../../scripts/app.js";

// Mirrors _MAGCACHE_PRESETS in comfyui-wanrt/nodes/tensorrt_perf.py -- keep these two in sync.
const MAGCACHE_PRESETS = {
    Fast: { magcache_thresh: 0.12, magcache_K: 4, magcache_retention_ratio: 0.15 },
    Balanced: { magcache_thresh: 0.06, magcache_K: 2, magcache_retention_ratio: 0.2 },
    Quality: { magcache_thresh: 0.03, magcache_K: 1, magcache_retention_ratio: 0.25 },
};

const TARGET_NODES = new Set(["TensorRTDiffusionLoader", "TensorRTCheckpointLoader"]);

app.registerExtension({
    name: "TensorRT-RT.MagCachePresetSync",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!TARGET_NODES.has(nodeData?.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const magcacheWidget = this.widgets?.find((w) => w.name === "MagCache");
            if (!magcacheWidget) return result;

            const syncFromPreset = (mode) => {
                const preset = MAGCACHE_PRESETS[mode];
                if (!preset) return; // "Disabled" / "Custom" -- leave whatever the user has alone
                for (const [name, value] of Object.entries(preset)) {
                    const w = this.widgets.find((w) => w.name === name);
                    if (w) w.value = value;
                }
            };

            const origCallback = magcacheWidget.callback;
            magcacheWidget.callback = (value, ...rest) => {
                syncFromPreset(value);
                return origCallback?.call(magcacheWidget, value, ...rest);
            };

            syncFromPreset(magcacheWidget.value); // reflect whatever preset is already selected on load

            return result;
        };
    },
});
