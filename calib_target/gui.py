"""One window to browse all extraction results: pick a bag from the list, see the whole cloud in grey
with the detected checkerboard coloured.  Started by view_results.py."""
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from .board import BoardSpec
from .viz import _outline_world

_NAMES = ("grey", "board", "outline", "axes", "sensor")


def find_results(results_dir):
    """Result folders that have both target.json and scene.ply, natural-sorted by name."""
    d = Path(results_dir)
    dirs = [p for p in d.iterdir() if p.is_dir() and (p / "target.json").exists() and (p / "scene.ply").exists()]
    return sorted(dirs, key=lambda p: (len(p.name), p.name))


class ResultsViewer:
    def __init__(self, results_dir, point_size=3.0, grey_voxel=0.01, width=1600, height=950):
        self.dirs = find_results(results_dir)
        if not self.dirs:
            raise SystemExit(f"No results with scene.ply found in {results_dir} - run main.py first.")
        self.grey_voxel, self.cache, self.current = grey_voxel, {}, None
        self.entries = {}
        for d in self.dirs:
            info = json.loads((d / "target.json").read_text())
            tag = "OK " if info.get("accepted") else "LOW"
            self.entries[f"{d.name}   [{tag} ncc {info.get('ncc')}]"] = d

        app = gui.Application.instance
        app.initialize()
        self.w = app.create_window("Checkerboard results - grey = scene, colour = detected board", width, height)
        em = self.w.theme.font_size
        self.em = em

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.w.renderer)
        self.scene.scene.set_background([0.06, 0.06, 0.08, 1.0])

        self.mat_pts = rendering.MaterialRecord()
        self.mat_pts.shader = "defaultUnlit"
        self.mat_pts.point_size = float(point_size)
        self.mat_line = rendering.MaterialRecord()
        self.mat_line.shader = "unlitLine"
        self.mat_line.line_width = 3.0
        self.mat_mesh = rendering.MaterialRecord()
        self.mat_mesh.shader = "defaultUnlit"

        pad = int(0.5 * em)
        self.panel = gui.Vert(pad, gui.Margins(pad, pad, pad, pad))
        self.panel.add_child(gui.Label("Point clouds (click one)"))
        self.list = gui.ListView()
        self.list.set_items(list(self.entries))
        self.list.set_on_selection_changed(self._on_select)
        self.panel.add_child(self.list)
        self.chk_grey = gui.Checkbox("Show grey scene")
        self.chk_grey.checked = True
        self.chk_grey.set_on_checked(lambda on: self._toggle("grey", on))
        self.chk_extra = gui.Checkbox("Show outline + axes")
        self.chk_extra.checked = True
        self.chk_extra.set_on_checked(self._toggle_extra)
        self.panel.add_child(self.chk_grey)
        self.panel.add_child(self.chk_extra)
        self.panel.add_child(gui.Label("Point size"))
        self.slider = gui.Slider(gui.Slider.DOUBLE)
        self.slider.set_limits(1.0, 10.0)
        self.slider.double_value = float(point_size)
        self.slider.set_on_value_changed(self._on_size)
        self.panel.add_child(self.slider)
        reset = gui.Button("Reset view")
        reset.set_on_clicked(self._reset_view)
        self.panel.add_child(reset)
        self.info = gui.Label("")
        self.panel.add_child(self.info)

        self.w.set_on_layout(self._layout)
        self.w.add_child(self.scene)
        self.w.add_child(self.panel)

        # start on the first accepted bag (else the first)
        first = next((k for k, d in self.entries.items() if "[OK" in k), next(iter(self.entries)))
        self.list.selected_index = list(self.entries).index(first)
        self._show(first)

    # ------------------------------------------------------------------ layout / callbacks
    def _layout(self, ctx):
        r = self.w.content_rect
        pw = int(24 * self.em)
        self.panel.frame = gui.Rect(r.x, r.y, pw, r.height)
        self.scene.frame = gui.Rect(r.x + pw, r.y, r.width - pw, r.height)

    def _on_select(self, value, is_double_click):
        self._show(value)

    def _toggle(self, name, on):
        if self.scene.scene.has_geometry(name):
            self.scene.scene.show_geometry(name, on)

    def _toggle_extra(self, on):
        for n in ("outline", "axes", "sensor"):
            self._toggle(n, on)

    def _on_size(self, value):
        self.mat_pts.point_size = float(value)
        for n in ("grey", "board"):
            if self.scene.scene.has_geometry(n):
                self.scene.scene.modify_geometry_material(n, self.mat_pts)

    # ------------------------------------------------------------------ data
    def _load(self, d):
        info = json.loads((d / "target.json").read_text())
        T = np.array(info["T_lidar_target"])
        spec = BoardSpec(info["board"]["cols"], info["board"]["rows"], info["board"]["square_m"])
        pcd = o3d.io.read_point_cloud(str(d / "scene.ply"))
        rgb, xyz = np.asarray(pcd.colors), np.asarray(pcd.points)
        is_board = np.abs(rgb[:, 0] - rgb[:, 1]) + np.abs(rgb[:, 1] - rgb[:, 2]) > 0.05
        grey = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[~is_board]))
        grey.colors = o3d.utility.Vector3dVector(rgb[~is_board])
        grey = grey.voxel_down_sample(self.grey_voxel)
        board = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[is_board]))
        board.colors = o3d.utility.Vector3dVector(rgb[is_board])
        out = _outline_world(T, spec)
        outline = o3d.geometry.LineSet(o3d.utility.Vector3dVector(out),
                                       o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(out) - 1)]))
        outline.paint_uniform_color([1, 1, 1])
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
        axes.transform(T)
        sensor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        return dict(info=info, T=T, grey=grey, board=board, outline=outline, axes=axes, sensor=sensor,
                    n_grey=len(xyz) - int(is_board.sum()), n_board=int(is_board.sum()))

    def _show(self, key):
        d = self.entries[key]
        if key not in self.cache:
            self.cache[key] = self._load(d)
        data = self.cache[key]
        sc = self.scene.scene
        for n in _NAMES:
            if sc.has_geometry(n):
                sc.remove_geometry(n)
        sc.add_geometry("grey", data["grey"], self.mat_pts)
        sc.add_geometry("board", data["board"], self.mat_pts)
        sc.add_geometry("outline", data["outline"], self.mat_line)
        sc.add_geometry("axes", data["axes"], self.mat_mesh)
        sc.add_geometry("sensor", data["sensor"], self.mat_mesh)
        self._toggle("grey", self.chk_grey.checked)
        self._toggle_extra(self.chk_extra.checked)
        self.current = key
        i = data["info"]
        self.info.text = (f"{d.name}\n"
                          f"accepted: {i.get('accepted')}   NCC {i.get('ncc')}\n"
                          f"board distance: {i.get('distance_m')} m\n"
                          f"square (fitted): {i.get('square_mm')} mm\n"
                          f"plane rms: {i.get('plane_rms_mm')} mm\n"
                          f"board points: {data['n_board']:,}\n"
                          f"scene points: {data['n_grey']:,}\n"
                          + (f"note: {i.get('note')}" if i.get("note") else ""))
        self._reset_view()

    def _reset_view(self):
        if self.current is None:
            return
        T = self.cache[self.current]["T"]
        o = T[:3, 3]
        bounds = self.scene.scene.bounding_box
        self.scene.setup_camera(60.0, bounds, o)
        eye = -0.6 * o / np.linalg.norm(o)                # a little behind the sensor, looking at the board
        self.scene.look_at(o, eye, T[:3, 1])


def run(results_dir, point_size=3.0, grey_voxel=0.01):
    viewer = ResultsViewer(results_dir, point_size, grey_voxel)   # noqa: F841 (keeps window alive)
    gui.Application.instance.run()
