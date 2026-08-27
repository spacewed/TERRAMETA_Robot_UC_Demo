#!/usr/bin/env python3
"""3D wireframe bounding box and skeleton renderer using moderngl.

Draws axis-aligned 3D bounding boxes as wireframe cubes in the scene,
using GL_LINES with a separate shader program. Each box is rendered as
12 line segments (24 vertices) forming a cube.

Also supports rendering 3D skeleton lines from pose estimation keypoints.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import moderngl
import numpy as np
import pygame

from depth_to_3d import PersonBBox3D
from yolo_pose_detector import COCO_SKELETON_CONNECTIONS

logger = logging.getLogger(__name__)

# COCO class id → (R, G, B, A) color map for per-class wireframe coloring.
# Colors are chosen to be visually distinct and readable on a dark background.
COCO_CLASS_COLORS: dict[int, tuple[float, float, float, float]] = {
    0:  (0.0, 1.0, 0.5, 1.0),    # person — green-cyan
    1:  (1.0, 0.6, 0.0, 1.0),    # bicycle — orange
    2:  (0.2, 0.6, 1.0, 1.0),    # car — blue
    3:  (1.0, 0.2, 0.8, 1.0),    # motorcycle — magenta
    4:  (0.6, 0.6, 1.0, 1.0),    # airplane — light blue
    5:  (0.0, 0.9, 0.7, 1.0),    # bus — teal
    6:  (1.0, 0.85, 0.0, 1.0),   # train — yellow
    7:  (0.8, 0.4, 0.0, 1.0),    # truck — brown-orange
    8:  (0.5, 1.0, 0.0, 1.0),    # boat — lime green
    9:  (1.0, 0.0, 0.0, 1.0),    # traffic light — red
    10: (0.7, 0.0, 1.0, 1.0),    # fire hydrant — purple
    11: (0.0, 0.8, 0.8, 1.0),    # stop sign — cyan
    12: (1.0, 0.5, 0.5, 1.0),    # parking meter — pink
    13: (0.9, 0.9, 0.0, 1.0),    # bench — yellow-green
    14: (0.0, 0.5, 1.0, 1.0),    # bird — sky blue
    15: (0.8, 0.0, 0.5, 1.0),    # cat — deep pink
    16: (0.0, 0.7, 0.3, 1.0),    # dog — forest green
    17: (0.9, 0.6, 0.9, 1.0),    # horse — lavender
    18: (0.5, 0.5, 0.0, 1.0),    # sheep — olive
    19: (0.3, 0.8, 0.0, 1.0),    # cow — bright green
    20: (1.0, 0.3, 0.0, 1.0),    # elephant — red-orange
    21: (0.0, 0.4, 0.8, 1.0),    # bear — dark blue
    22: (0.6, 0.0, 0.0, 1.0),    # zebra — dark red
    23: (0.0, 0.6, 0.6, 1.0),    # giraffe — dark teal
    24: (0.8, 0.8, 0.4, 1.0),    # backpack — khaki
    25: (0.4, 0.0, 0.8, 1.0),    # umbrella — indigo
    26: (0.9, 0.4, 0.7, 1.0),    # handbag — rose
    27: (0.6, 0.3, 0.0, 1.0),    # tie — brown
    28: (0.0, 0.3, 0.0, 1.0),    # suitcase — dark green
    29: (0.7, 0.7, 0.7, 1.0),    # frisbee — gray
    30: (1.0, 0.7, 0.3, 1.0),    # skis — peach
    31: (0.3, 0.0, 0.5, 1.0),    # snowboard — deep purple
    32: (0.0, 0.9, 0.9, 1.0),    # sports ball — aqua
    33: (0.8, 0.0, 0.0, 1.0),    # kite — crimson
    34: (0.5, 0.0, 0.0, 1.0),    # baseball bat — maroon
    35: (0.0, 0.5, 0.5, 1.0),    # baseball glove — teal
    36: (0.7, 0.5, 0.0, 1.0),    # skateboard — tan
    37: (0.0, 0.0, 0.7, 1.0),    # surfboard — navy
    38: (0.9, 0.0, 0.3, 1.0),    # tennis racket — ruby
    39: (0.0, 0.8, 0.4, 1.0),    # bottle — emerald
    40: (0.6, 0.0, 0.6, 1.0),    # wine glass — violet
    41: (0.0, 0.0, 0.0, 1.0),    # cup — black
    42: (0.8, 0.8, 0.0, 1.0),    # fork — olive-yellow
    43: (0.0, 0.6, 0.0, 1.0),    # knife — green
    44: (0.5, 0.0, 0.5, 1.0),    # spoon — purple
    45: (0.9, 0.3, 0.3, 1.0),    # bowl — salmon
    46: (0.0, 0.0, 0.5, 1.0),    # banana — navy-blue
    47: (0.0, 0.5, 0.0, 1.0),    # apple — dark green
    48: (0.8, 0.0, 0.8, 1.0),    # sandwich — fuchsia
    49: (0.5, 0.3, 0.0, 1.0),    # orange — sienna
    50: (0.0, 0.3, 0.0, 1.0),    # broccoli — forest green
    51: (0.0, 0.0, 0.3, 1.0),    # carrot — dark navy
    52: (0.6, 0.0, 0.3, 1.0),    # hot dog — burgundy
    53: (0.0, 0.4, 0.4, 1.0),    # pizza — dark cyan
    54: (0.7, 0.0, 0.7, 1.0),    # donut — plum
    55: (0.0, 0.7, 0.7, 1.0),    # cake — turquoise
    56: (0.4, 0.4, 0.0, 1.0),    # chair — olive
    57: (0.0, 0.0, 0.4, 1.0),    # couch — midnight blue
    58: (0.5, 0.5, 0.5, 1.0),    # potted plant — gray
    59: (0.8, 0.5, 0.0, 1.0),    # bed — amber
    60: (0.0, 0.3, 0.6, 1.0),    # dining table — steel blue
    61: (0.6, 0.0, 0.4, 1.0),    # toilet — plum-red
    62: (0.0, 0.6, 0.3, 1.0),    # tv — sea green
    63: (0.3, 0.0, 0.3, 1.0),    # laptop — dark violet
    64: (0.0, 0.0, 0.6, 1.0),    # mouse — royal blue
    65: (0.6, 0.6, 0.0, 1.0),    # remote — olive-yellow
    66: (0.0, 0.4, 0.0, 1.0),    # keyboard — dark green
    67: (0.0, 0.0, 0.3, 1.0),    # cell phone — dark blue
    68: (0.5, 0.0, 0.3, 1.0),    # microwave — wine
    69: (0.0, 0.5, 0.3, 1.0),    # oven — jade
    70: (0.3, 0.3, 0.0, 1.0),    # toaster — olive-drab
    71: (0.0, 0.0, 0.2, 1.0),    # sink — very dark blue
    72: (0.4, 0.0, 0.0, 1.0),    # refrigerator — dark red
    73: (0.0, 0.3, 0.3, 1.0),    # book — dark teal
    74: (0.5, 0.5, 0.5, 1.0),    # clock — gray
    75: (0.0, 0.0, 0.0, 1.0),    # vase — black
    76: (0.7, 0.0, 0.0, 1.0),    # scissors — red
    77: (0.0, 0.0, 0.0, 1.0),    # teddy bear — black
    78: (0.0, 0.0, 0.0, 1.0),    # hair drier — black
    79: (0.0, 0.0, 0.0, 1.0),    # toothbrush — black
}

# Default color for unknown class ids
DEFAULT_BBOX_COLOR = (0.0, 1.0, 0.5, 1.0)  # Green-cyan
VLM_BBOX_COLOR = (0.95, 0.78, 0.12, 1.0)  # Amber semantic VLM boxes


def get_class_color(class_id: int) -> tuple[float, float, float, float]:
    """Get the RGBA color for a COCO class id."""
    return COCO_CLASS_COLORS.get(class_id, DEFAULT_BBOX_COLOR)


def get_box_color(box: PersonBBox3D) -> tuple[float, float, float, float]:
    """Get a stable wireframe color for a rendered box."""
    if box.source.startswith("vlm"):
        return VLM_BBOX_COLOR
    return get_class_color(box.class_id)

# Vertex shader: transforms 3D positions through view/projection matrices.
# Supports per-vertex color via in_color attribute.
VERTEX_SHADER = """
#version 330
in vec3 in_position;
in vec4 in_color;
uniform mat4 view;
uniform mat4 projection;
out vec4 v_color;
void main() {
    gl_Position = projection * view * vec4(in_position, 1.0);
    v_color = in_color;
}
"""

# Fragment shader: outputs per-vertex color for each line segment.
FRAGMENT_SHADER = """
#version 330
in vec4 v_color;
out vec4 color;
void main() {
    color = v_color;
}
"""

TEXT_VERTEX_SHADER = """
#version 330
in vec2 in_position;
in vec2 in_uv;
out vec2 frag_uv;
void main() {
    frag_uv = in_uv;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

TEXT_FRAGMENT_SHADER = """
#version 330
uniform sampler2D label_tex;
in vec2 frag_uv;
out vec4 color;
void main() {
    color = texture(label_tex, frag_uv);
}
"""

LABEL_PADDING_PX = 5
LABEL_FONT_SIZE_PX = 18
LABEL_VIEWPORT_MARGIN_PX = 4
LABEL_ANCHOR_GAP_PX = 6
LABEL_ANCHOR_REPOSITION_DISTANCE_M = 0.35
LABEL_TEXT_REFRESH_INTERVAL_S = 0.5
LABEL_TRACK_MATCH_MAX_DISTANCE_M = 1.5


@dataclass
class _BoxLabel:
    texture: moderngl.Texture
    size_px: Tuple[int, int]
    text: str
    text_updated_s: float
    camera_id: int
    label: str
    source: str
    anchor_world: np.ndarray
    center_world: np.ndarray
    class_id: int = 0  # COCO class id for per-class color coding


def _box_wireframe_vertices(
    center_xyz: np.ndarray,
    size_xyz: np.ndarray,
) -> np.ndarray:
    """Generate 24 vertices (12 edges) for a wireframe axis-aligned box.

    Args:
        center_xyz: [cx, cy, cz] centre of the box in world coords.
        size_xyz: [width, height, depth] extents in metres.

    Returns:
        np.ndarray of shape (24, 3), dtype float32. Each pair of consecutive
        vertices forms one edge of the wireframe cube.
    """
    cx, cy, cz = center_xyz
    hw, hh, hd = size_xyz / 2.0

    # 8 corners of the box
    corners = np.array(
        [
            [cx - hw, cy - hh, cz - hd],  # 0: left-bottom-back
            [cx + hw, cy - hh, cz - hd],  # 1: right-bottom-back
            [cx + hw, cy + hh, cz - hd],  # 2: right-top-back
            [cx - hw, cy + hh, cz - hd],  # 3: left-top-back
            [cx - hw, cy - hh, cz + hd],  # 4: left-bottom-front
            [cx + hw, cy - hh, cz + hd],  # 5: right-bottom-front
            [cx + hw, cy + hh, cz + hd],  # 6: right-top-front
            [cx - hw, cy + hh, cz + hd],  # 7: left-top-front
        ],
        dtype=np.float32,
    )

    # 12 edges as pairs of corner indices
    edge_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # back face
        (4, 5), (5, 6), (6, 7), (7, 4),  # front face
        (0, 4), (1, 5), (2, 6), (3, 7),  # connecting edges
    ]

    vertices = []
    for i, j in edge_pairs:
        vertices.append(corners[i])
        vertices.append(corners[j])

    return np.array(vertices, dtype=np.float32)


class BBoxRenderer:
    """Renders 3D wireframe bounding boxes and skeletons via moderngl.

    Usage:
        renderer = BBoxRenderer(ctx, window_size)
        renderer.update_boxes(detections)
        renderer.update_skeletons(pose_detections)
        renderer.render(view_matrix, projection_matrix)
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        window_size: Tuple[int, int],
    ) -> None:
        self._ctx = ctx
        self._window_size = window_size
        self._program: Optional[moderngl.Program] = None
        self._vbo: Optional[moderngl.Buffer] = None
        self._vao: Optional[moderngl.VertexArray] = None
        self._view_uniform = None
        self._projection_uniform = None
        self._vertex_count = 0
        self._font: Optional[pygame.font.Font] = None
        self._text_program: Optional[moderngl.Program] = None
        self._text_texture_uniform = None
        self._text_vbo: Optional[moderngl.Buffer] = None
        self._text_vao: Optional[moderngl.VertexArray] = None
        self._labels: List[_BoxLabel] = []
        self._box_signature: Tuple[tuple, ...] = ()
        self._initialized = False

        # Skeleton rendering state
        self._skeleton_vbo: Optional[moderngl.Buffer] = None
        self._skeleton_vao: Optional[moderngl.VertexArray] = None
        self._skeleton_vertex_count = 0
        self._skeleton_signature: Tuple[tuple, ...] = ()

        self._init_shader()

    def _init_shader(self) -> None:
        """Create the shader program and uniforms."""
        try:
            self._program = self._ctx.program(
                vertex_shader=VERTEX_SHADER,
                fragment_shader=FRAGMENT_SHADER,
            )
            self._view_uniform = self._program["view"]
            self._projection_uniform = self._program["projection"]
            self._init_text_renderer()
            self._initialized = True
        except Exception as exc:
            logger.error("Failed to initialize BBox shader: %s", exc)
            self._initialized = False

    def _init_text_renderer(self) -> None:
        if not pygame.font.get_init():
            pygame.font.init()
        self._font = pygame.font.Font(None, LABEL_FONT_SIZE_PX)
        self._text_program = self._ctx.program(
            vertex_shader=TEXT_VERTEX_SHADER,
            fragment_shader=TEXT_FRAGMENT_SHADER,
        )
        self._text_texture_uniform = self._text_program["label_tex"]
        self._text_texture_uniform.value = 0
        self._text_vbo = self._ctx.buffer(reserve=6 * 4 * 4, dynamic=True)
        self._text_vao = self._ctx.vertex_array(
            self._text_program,
            [(self._text_vbo, "2f 2f", "in_position", "in_uv")],
        )

    def update_boxes(self, boxes: List[PersonBBox3D]) -> None:
        """Rebuild VBO with current set of 3D bounding boxes.

        Args:
            boxes: List of PersonBBox3D objects to render.
        """
        if not self._initialized or self._program is None:
            return

        box_signature = self._make_box_signature(boxes)
        if (
            box_signature == self._box_signature
            and (not boxes or (self._vao is not None and self._vertex_count > 0))
        ):
            return

        if not boxes:
            # Release existing buffers and set count to 0
            self._release_wireframe_buffers()
            self._release_label_textures()
            self._box_signature = ()
            self._vertex_count = 0
            return

        # Generate all wireframe vertices with per-class colors
        all_vertices = []
        all_colors = []
        for box in boxes:
            verts = _box_wireframe_vertices(box.center_xyz, box.size_xyz)
            all_vertices.append(verts)
            color = get_box_color(box)
            color_arr = np.full((len(verts), 4), color, dtype=np.float32)
            all_colors.append(color_arr)

        combined_positions = np.concatenate(all_vertices, axis=0)
        combined_colors = np.concatenate(all_colors, axis=0)

        # Interleave position (3f) + color (4f) into a single VBO
        combined_data = np.hstack([combined_positions, combined_colors]).astype(np.float32)

        # Release old buffers if they exist
        self._release_wireframe_buffers()

        # Create new buffers with interleaved position + color
        self._vbo = self._ctx.buffer(combined_data.tobytes())
        self._vao = self._ctx.vertex_array(
            self._program, [(self._vbo, "3f 4f", "in_position", "in_color")]
        )
        self._vertex_count = len(combined_positions)
        self._update_labels(boxes)
        self._box_signature = box_signature

    def render(
        self,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
    ) -> None:
        """Draw all bounding boxes with per-class colors.

        Args:
            view_matrix: 4x4 view matrix as numpy array.
            projection_matrix: 4x4 projection matrix as numpy array.
        """
        if (
            not self._initialized
            or self._vao is None
            or self._vertex_count == 0
        ):
            return

        try:
            self._view_uniform.write(view_matrix.astype("f4").tobytes())
            self._projection_uniform.write(projection_matrix.astype("f4").tobytes())

            # Disable depth test so boxes are always visible on top of point cloud
            self._ctx.disable(moderngl.DEPTH_TEST)
            self._vao.render(moderngl.LINES, vertices=self._vertex_count)
            self._render_labels(view_matrix, projection_matrix)
        except Exception as exc:
            logger.warning("Failed to render bounding boxes: %s", exc)
        finally:
            self._ctx.disable(moderngl.BLEND)
            self._ctx.enable(moderngl.DEPTH_TEST)

    def release(self) -> None:
        """Release all GPU resources."""
        self._release_wireframe_buffers()
        self._release_skeleton_buffers()
        self._release_label_textures()
        if self._program is not None:
            try:
                self._program.release()
            except Exception:
                pass
            self._program = None
        self._view_uniform = None
        self._projection_uniform = None
        for buf in (self._text_vao, self._text_vbo):
            if buf is not None:
                try:
                    buf.release()
                except Exception:
                    pass
        if self._text_program is not None:
            try:
                self._text_program.release()
            except Exception:
                pass
        self._text_program = None
        self._text_texture_uniform = None
        self._text_vao = None
        self._text_vbo = None
        self._font = None
        self._initialized = False

    def _release_wireframe_buffers(self) -> None:
        """Release VBO and VAO if they exist."""
        for buf in (self._vao, self._vbo):
            if buf is not None:
                try:
                    buf.release()
                except Exception:
                    pass
        self._vao = None
        self._vbo = None
        self._vertex_count = 0

    def _release_skeleton_buffers(self) -> None:
        """Release skeleton VBO and VAO if they exist."""
        for buf in (self._skeleton_vao, self._skeleton_vbo):
            if buf is not None:
                try:
                    buf.release()
                except Exception:
                    pass
        self._skeleton_vao = None
        self._skeleton_vbo = None
        self._skeleton_vertex_count = 0

    def _release_label_textures(self) -> None:
        for label in self._labels:
            self._release_label_texture(label)
        self._labels = []

    def _update_labels(self, boxes: List[PersonBBox3D]) -> None:
        if self._font is None:
            self._release_label_textures()
            return

        previous_labels = self._labels
        next_labels: List[_BoxLabel] = []
        matched_labels = set()
        now = time.monotonic()

        for box in boxes:
            match_index = self._find_matching_label_index(
                previous_labels,
                matched_labels,
                box,
            )
            if match_index is None:
                next_labels.append(self._new_label(box, now))
                continue

            label = previous_labels[match_index]
            matched_labels.add(match_index)
            self._update_tracked_label(label, box, now)
            next_labels.append(label)

        for index, label in enumerate(previous_labels):
            if index not in matched_labels:
                self._release_label_texture(label)
        self._labels = next_labels

    def _new_label(self, box: PersonBBox3D, now: float) -> _BoxLabel:
        text = self._format_box_label(box)
        color = get_box_color(box)
        texture, size_px = self._build_label_texture(text, border_color=color)
        return _BoxLabel(
            texture=texture,
            size_px=size_px,
            text=text,
            text_updated_s=now,
            camera_id=int(box.camera_id),
            label=box.label,
            source=box.source,
            anchor_world=self._label_anchor_for_box(box),
            center_world=np.asarray(box.center_xyz, dtype=np.float32).copy(),
            class_id=int(box.class_id),
        )

    def _update_tracked_label(
        self,
        label: _BoxLabel,
        box: PersonBBox3D,
        now: float,
    ) -> None:
        label.anchor_world = self._stabilize_label_anchor(
            label.anchor_world,
            self._label_anchor_for_box(box),
        )
        label.center_world = np.asarray(box.center_xyz, dtype=np.float32).copy()
        label.class_id = int(box.class_id)

        text = self._format_box_label(box)
        if (
            text == label.text
            or now - label.text_updated_s < LABEL_TEXT_REFRESH_INTERVAL_S
        ):
            return

        color = get_box_color(box)
        texture, size_px = self._build_label_texture(text, border_color=color)
        self._release_label_texture(label)
        label.texture = texture
        label.size_px = size_px
        label.text = text
        label.text_updated_s = now

    @staticmethod
    def _label_anchor_for_box(box: PersonBBox3D) -> np.ndarray:
        anchor_world = np.asarray(box.center_xyz, dtype=np.float32).copy()
        anchor_world[1] += float(box.size_xyz[1]) / 2.0
        return anchor_world

    @staticmethod
    def _stabilize_label_anchor(
        previous_anchor_world: np.ndarray,
        target_anchor_world: np.ndarray,
    ) -> np.ndarray:
        previous = np.asarray(previous_anchor_world, dtype=np.float32)
        target = np.asarray(target_anchor_world, dtype=np.float32)
        if np.linalg.norm(target - previous) < LABEL_ANCHOR_REPOSITION_DISTANCE_M:
            return previous
        return target

    @staticmethod
    def _find_matching_label_index(
        labels: List[_BoxLabel],
        matched_labels: set,
        box: PersonBBox3D,
    ) -> Optional[int]:
        center_world = np.asarray(box.center_xyz, dtype=np.float32)
        best_index = None
        best_distance = LABEL_TRACK_MATCH_MAX_DISTANCE_M

        for index, label in enumerate(labels):
            if index in matched_labels:
                continue
            if (
                label.camera_id != int(box.camera_id)
                or label.label != box.label
                or label.source != box.source
            ):
                continue

            distance = float(np.linalg.norm(label.center_world - center_world))
            if distance <= best_distance:
                best_index = index
                best_distance = distance

        return best_index

    @staticmethod
    def _release_label_texture(label: _BoxLabel) -> None:
        try:
            label.texture.release()
        except Exception:
            pass

    def _build_label_texture(
        self,
        text: str,
        border_color: tuple = (0.0, 1.0, 0.5, 1.0),
    ) -> Tuple[moderngl.Texture, Tuple[int, int]]:
        if self._font is None:
            raise RuntimeError("bbox label font is unavailable")

        text_surface = self._font.render(text, True, (240, 255, 248))
        width = text_surface.get_width() + (LABEL_PADDING_PX * 2)
        height = text_surface.get_height() + (LABEL_PADDING_PX * 2)
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(
            surface,
            (0, 0, 0, 190),
            surface.get_rect(),
            border_radius=3,
        )
        # Convert RGBA float tuple to 0-255 int tuple for pygame
        border_rgba = (
            int(border_color[0] * 255),
            int(border_color[1] * 255),
            int(border_color[2] * 255),
            int(border_color[3] * 255) if len(border_color) > 3 else 220,
        )
        pygame.draw.rect(
            surface,
            border_rgba,
            surface.get_rect(),
            width=1,
            border_radius=3,
        )
        surface.blit(text_surface, (LABEL_PADDING_PX, LABEL_PADDING_PX))

        texture = self._ctx.texture(
            (width, height),
            4,
            pygame.image.tobytes(surface, "RGBA", True),
            alignment=1,
        )
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False
        return texture, (width, height)

    def _render_labels(
        self,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
    ) -> None:
        if self._text_vao is None or self._text_vbo is None:
            return

        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        for label in self._labels:
            vertices = self._label_quad_vertices(
                label,
                view_matrix,
                projection_matrix,
            )
            if vertices is None:
                continue
            label.texture.use(location=0)
            self._text_vbo.write(vertices.tobytes())
            self._text_vao.render(moderngl.TRIANGLES, vertices=6)
        self._ctx.disable(moderngl.BLEND)

    def _label_quad_vertices(
        self,
        label: _BoxLabel,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
    ) -> Optional[np.ndarray]:
        screen_anchor = self._project_world_to_screen(
            label.anchor_world,
            view_matrix,
            projection_matrix,
            self._window_size,
        )
        if screen_anchor is None:
            return None

        viewport_w, viewport_h = self._window_size
        label_w, label_h = label.size_px
        left = np.clip(
            screen_anchor[0],
            LABEL_VIEWPORT_MARGIN_PX,
            max(LABEL_VIEWPORT_MARGIN_PX, viewport_w - label_w - LABEL_VIEWPORT_MARGIN_PX),
        )
        top = np.clip(
            screen_anchor[1] - label_h - LABEL_ANCHOR_GAP_PX,
            LABEL_VIEWPORT_MARGIN_PX,
            max(LABEL_VIEWPORT_MARGIN_PX, viewport_h - label_h - LABEL_VIEWPORT_MARGIN_PX),
        )
        right = left + label_w
        bottom = top + label_h
        x1 = (2.0 * left / viewport_w) - 1.0
        x2 = (2.0 * right / viewport_w) - 1.0
        y1 = 1.0 - (2.0 * top / viewport_h)
        y2 = 1.0 - (2.0 * bottom / viewport_h)
        return np.array(
            [
                [x1, y1, 0.0, 1.0],
                [x2, y1, 1.0, 1.0],
                [x2, y2, 1.0, 0.0],
                [x1, y1, 0.0, 1.0],
                [x2, y2, 1.0, 0.0],
                [x1, y2, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _project_world_to_screen(
        position_xyz: np.ndarray,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        window_size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        position = np.array(
            [position_xyz[0], position_xyz[1], position_xyz[2], 1.0],
            dtype=np.float32,
        )
        clip = position @ view_matrix @ projection_matrix
        if clip[3] <= 0.0:
            return None

        ndc = clip[:3] / clip[3]
        if ndc[2] < -1.0 or ndc[2] > 1.0:
            return None

        viewport_w, viewport_h = window_size
        return np.array(
            [
                (ndc[0] + 1.0) * viewport_w / 2.0,
                (1.0 - ndc[1]) * viewport_h / 2.0,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _format_box_label(box: PersonBBox3D) -> str:
        x_m, y_m, z_m = [float(value) for value in box.size_xyz]
        prefix = "VLM" if box.source.startswith("vlm") else "Detected"
        return (
            f"{prefix} {box.label} {box.confidence * 100.0:.0f}% | "
            f"dims X {x_m:.2f}m Y {y_m:.2f}m Z {z_m:.2f}m"
        )

    @staticmethod
    def _make_box_signature(boxes: List[PersonBBox3D]) -> Tuple[tuple, ...]:
        return tuple(
            (
                box.camera_id,
                box.label,
                box.source,
                int(box.class_id),
                round(float(box.confidence), 4),
                round(float(box.timestamp), 6),
                tuple(np.asarray(box.center_xyz, dtype=np.float32)),
                tuple(np.asarray(box.size_xyz, dtype=np.float32)),
            )
            for box in boxes
        )

    def update_skeletons(self, poses: list) -> None:
        """Rebuild VBO with current set of 3D skeleton lines.

        Args:
            poses: List of PersonPose3D objects with 3D keypoints.
        """
        if not self._initialized or self._program is None:
            return

        skeleton_signature = self._make_skeleton_signature(poses)
        if (
            skeleton_signature == self._skeleton_signature
            and (not poses or (self._skeleton_vao is not None and self._skeleton_vertex_count > 0))
        ):
            return

        if not poses:
            self._release_skeleton_buffers()
            self._skeleton_signature = ()
            return

        # Generate skeleton line vertices for each pose (always person class)
        all_vertices = []
        for pose in poses:
            verts = _skeleton_vertices(pose.keypoints_3d, pose.keypoints_confidence)
            if len(verts) > 0:
                all_vertices.append(verts)

        if not all_vertices:
            self._release_skeleton_buffers()
            self._skeleton_signature = skeleton_signature
            return

        combined_positions = np.concatenate(all_vertices, axis=0)

        # Skeletons are always person class — use bright yellow for visibility
        skeleton_color = (1.0, 0.9, 0.2, 1.0)  # Bright yellow
        combined_colors = np.full((len(combined_positions), 4), skeleton_color, dtype=np.float32)

        # Interleave position (3f) + color (4f) into a single VBO
        combined_data = np.hstack([combined_positions, combined_colors]).astype(np.float32)

        self._release_skeleton_buffers()

        self._skeleton_vbo = self._ctx.buffer(combined_data.tobytes())
        self._skeleton_vao = self._ctx.vertex_array(
            self._program, [(self._skeleton_vbo, "3f 4f", "in_position", "in_color")]
        )
        self._skeleton_vertex_count = len(combined_positions)
        self._skeleton_signature = skeleton_signature

    def render_skeletons(
        self,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
    ) -> None:
        """Draw all skeleton lines.

        Args:
            view_matrix: 4x4 view matrix as numpy array.
            projection_matrix: 4x4 projection matrix as numpy array.
        """
        if (
            not self._initialized
            or self._skeleton_vao is None
            or self._skeleton_vertex_count == 0
        ):
            return

        try:
            self._view_uniform.write(view_matrix.astype("f4").tobytes())
            self._projection_uniform.write(projection_matrix.astype("f4").tobytes())

            self._ctx.disable(moderngl.DEPTH_TEST)
            self._ctx.enable(moderngl.BLEND)
            self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self._skeleton_vao.render(moderngl.TRIANGLES, vertices=self._skeleton_vertex_count)
        except Exception as exc:
            logger.warning("Failed to render skeletons: %s", exc)
        finally:
            self._ctx.disable(moderngl.BLEND)
            self._ctx.enable(moderngl.DEPTH_TEST)

    @staticmethod
    def _make_skeleton_signature(poses: list) -> Tuple[tuple, ...]:
        return tuple(
            (
                pose.camera_id,
                pose.label,
                round(float(pose.confidence), 4),
                round(float(pose.timestamp), 6),
                np.asarray(pose.center_xyz, dtype=np.float32).tobytes(),
                np.asarray(pose.size_xyz, dtype=np.float32).tobytes(),
                # Byte signatures keep invalid NaN keypoints cacheable.
                np.asarray(pose.keypoints_3d, dtype=np.float32).tobytes(),
                np.asarray(pose.keypoints_confidence, dtype=np.float32).tobytes(),
            )
            for pose in poses
        )


def _skeleton_bone_quad_vertices(
    p_start: np.ndarray,
    p_end: np.ndarray,
    half_width: float = 0.02,
) -> np.ndarray:
    """Generate 6 vertices (2 triangles) for a thick bone segment between two 3D points.

    Computes a perpendicular offset to create a screen-aligned quad.
    Handles all orientations by trying multiple reference axes.

    Args:
        p_start: [X, Y, Z] start point in world coords.
        p_end: [X, Y, Z] end point in world coords.
        half_width: Half-width of the bone in metres.

    Returns:
        np.ndarray of shape (6, 3), dtype float32 — two triangles forming a quad.
    """
    direction = p_end - p_start
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return np.zeros((0, 3), dtype=np.float32)

    dir_normalized = direction / length

    # Try multiple reference axes to find a valid perpendicular
    ref_axes = [
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    ]

    right = None
    for ref in ref_axes:
        candidate = np.cross(dir_normalized, ref)
        candidate_len = np.linalg.norm(candidate)
        if candidate_len > 1e-6:
            right = candidate / candidate_len * half_width
            break

    if right is None:
        return np.zeros((0, 3), dtype=np.float32)

    # Four corners of the quad
    v0 = p_start - right
    v1 = p_start + right
    v2 = p_end + right
    v3 = p_end - right

    # Two triangles: (v0, v1, v2) and (v0, v2, v3)
    return np.array([
        v0, v1, v2,
        v0, v2, v3,
    ], dtype=np.float32)


def _skeleton_keypoint_vertices(
    keypoints_3d: np.ndarray,
    keypoints_confidence: np.ndarray,
    radius: float = 0.03,
    confidence_threshold: float = 0.5,
) -> np.ndarray:
    """Generate diamond-shaped vertices for each valid keypoint.

    Each keypoint is rendered as a small diamond (8 triangles = 24 vertices).

    Args:
        keypoints_3d: shape (K, 3) — [X, Y, Z] per keypoint in world coords.
        keypoints_confidence: shape (K,) — confidence per keypoint.
        radius: Radius of the keypoint marker in metres.
        confidence_threshold: Minimum confidence to include a keypoint.

    Returns:
        np.ndarray of shape (N*24, 3), dtype float32 — triangles for each keypoint.
    """
    all_vertices = []
    for k in range(len(keypoints_3d)):
        if keypoints_confidence[k] < confidence_threshold:
            continue
        p = keypoints_3d[k]
        if np.isnan(p).any():
            continue

        cx, cy, cz = p
        r = radius

        # Diamond shape: 8 triangles forming an octahedron
        top = np.array([cx, cy + r, cz], dtype=np.float32)
        bottom = np.array([cx, cy - r, cz], dtype=np.float32)
        left = np.array([cx - r, cy, cz], dtype=np.float32)
        right = np.array([cx + r, cy, cz], dtype=np.float32)
        front = np.array([cx, cy, cz + r], dtype=np.float32)
        back = np.array([cx, cy, cz - r], dtype=np.float32)
        center = np.array([cx, cy, cz], dtype=np.float32)

        all_vertices.extend([
            center, top, right,
            center, right, bottom,
            center, bottom, left,
            center, left, top,
            center, top, front,
            center, front, bottom,
            center, bottom, back,
            center, back, top,
        ])

    if not all_vertices:
        return np.zeros((0, 3), dtype=np.float32)

    return np.array(all_vertices, dtype=np.float32)


def _skeleton_vertices(
    keypoints_3d: np.ndarray,
    keypoints_confidence: np.ndarray,
    confidence_threshold: float = 0.5,
    bone_half_width: float = 0.02,
    keypoint_radius: float = 0.03,
) -> np.ndarray:
    """Generate triangle vertices for a 3D skeleton (bones + keypoints).

    Each bone is rendered as a quad (2 triangles) and each keypoint as a
    diamond (8 triangles). All rendered as GL_TRIANGLES.

    Args:
        keypoints_3d: shape (K, 3) — [X, Y, Z] per keypoint in world coords.
        keypoints_confidence: shape (K,) — confidence per keypoint.
        confidence_threshold: Minimum confidence to include a keypoint.
        bone_half_width: Half-width of bone segments in metres.
        keypoint_radius: Radius of keypoint markers in metres.

    Returns:
        np.ndarray of shape (N, 3), dtype float32 — triangle vertices.
    """
    all_vertices = []

    # Generate bone quads
    for i, j in COCO_SKELETON_CONNECTIONS:
        if i >= len(keypoints_3d) or j >= len(keypoints_3d):
            continue
        if keypoints_confidence[i] < confidence_threshold or keypoints_confidence[j] < confidence_threshold:
            continue
        p_i = keypoints_3d[i]
        p_j = keypoints_3d[j]
        if np.isnan(p_i).any() or np.isnan(p_j).any():
            continue
        quad_verts = _skeleton_bone_quad_vertices(p_i, p_j, half_width=bone_half_width)
        if len(quad_verts) > 0:
            all_vertices.append(quad_verts)

    # Generate keypoint diamonds
    kpt_verts = _skeleton_keypoint_vertices(
        keypoints_3d, keypoints_confidence,
        radius=keypoint_radius,
        confidence_threshold=confidence_threshold,
    )
    if len(kpt_verts) > 0:
        all_vertices.append(kpt_verts)

    if not all_vertices:
        return np.zeros((0, 3), dtype=np.float32)

    return np.concatenate(all_vertices, axis=0)
