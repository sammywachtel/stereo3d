"""Blender compositing script for stereo 3D refinement.

Run with: blender -b -P blender/stereo_composite.py

This is a starting point — extend it for convergence adjustments,
depth-based fog, titles, or whatever post-processing you need.
"""

import bpy
import os
from pathlib import Path


def setup_compositor(
    stereo_dir: str = "work/stereo_frames",
    output_dir: str = "work/preview",
):
    """Configure Blender's compositor for SBS stereo post-processing.

    Currently sets up a pass-through with an example color correction
    node. Add your own compositing nodes as needed.
    """
    stereo_dir = Path(stereo_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Enable compositing
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    # Clear default nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Image input — load the first stereo frame as a template
    frames = sorted(stereo_dir.glob("frame_*.png"))
    if not frames:
        print(f"No stereo frames found in {stereo_dir}")
        return

    img_node = tree.nodes.new("CompositorNodeImage")
    img = bpy.data.images.load(str(frames[0]))
    img_node.image = img
    img_node.location = (0, 300)

    # Color balance — subtle warm shift, tweak to taste
    color_node = tree.nodes.new("CompositorNodeColorBalance")
    color_node.correction_method = "LIFT_GAMMA_GAIN"
    color_node.location = (300, 300)

    # Output
    output_node = tree.nodes.new("CompositorNodeComposite")
    output_node.location = (600, 300)

    # Wire it up
    links = tree.links
    links.new(img_node.outputs[0], color_node.inputs[1])
    links.new(color_node.outputs[0], output_node.inputs[0])

    # Configure output
    scene = bpy.context.scene
    scene.render.filepath = str(output_dir / "preview_")
    scene.render.image_settings.file_format = "PNG"

    print(f"Compositor configured. {len(frames)} frames available.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    setup_compositor()
    print("Blender compositor ready. Render with: bpy.ops.render.render()")
