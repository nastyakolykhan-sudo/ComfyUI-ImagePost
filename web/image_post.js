import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Shows the text these nodes return in ui.text (residual table, report.json) in a read-only box on the node.
const TEXT_NODES = new Set(["ImagePostFitWarp", "ImagePostQASheet", "ImagePostSavePSD"]);

app.registerExtension({
    name: "ImagePost.ReportText",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!TEXT_NODES.has(nodeData.name)) {
            return;
        }
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const text = message?.text;
            if (text === undefined) {
                return;
            }
            let widget = this.widgets?.find((w) => w.name === "image_post_text");
            if (!widget) {
                widget = ComfyWidgets["STRING"](this, "image_post_text", ["STRING", { multiline: true }], app).widget;
                widget.options = { ...(widget.options ?? {}), serialize: false };  // display only: not a node input
                widget.inputEl.readOnly = true;
                widget.inputEl.style.fontFamily = "monospace";
                widget.inputEl.style.fontSize = "11px";
            }
            widget.value = Array.isArray(text) ? text.join("\n") : String(text);
            this.setSize?.([Math.max(this.size[0], 520), Math.max(this.size[1], 260)]);
            app.graph?.setDirtyCanvas(true, true);
        };
    },
});
