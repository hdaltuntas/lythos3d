"""Unstructured meshes of a site with gmsh's OpenCASCADE kernel.

The geometry is built the way the ground is described:

1. a box over the plan extent, from the model base to above the highest
   ground;
2. the ground surface, extruded upwards, is cut away from the box;
3. the soil interfaces cut what remains into layers;
4. each excavation lift - its plan polygon between two levels - is clipped to
   the ground and cut into it.

Every surface is a loft through splines sampled from the
:class:`~lythos3d.core.site.SoilProfile`, so a plane (one borehole, or
boreholes that agree) is reproduced exactly and an interpolated surface
closely.  After meshing, each gmsh volume is labelled with the soil and the
lift that most of its elements' centroids fall in: a volume's own centroid
can lie outside it (a ring of soil round a pit has its centroid in the pit),
and the vote also absorbs the small difference between the lofted surfaces
and the interpolated ones.

gmsh is only imported here, so the rest of Lythos 3D works without it.
"""

from __future__ import annotations

import numpy as np

from typing import NamedTuple

from .mesh import Mesh, quadratic_from_linear
from .site import Excavation, SiteWall, SoilProfile


class SiteMesh(NamedTuple):
    """What :func:`mesh_site` produces."""

    mesh: Mesh
    #: lift name -> element mask
    groups: dict
    #: wall name -> 6-node faces (nf, 6) of the mesh on that wall
    wall_faces: dict
    #: node index of each point asked to be a node, in order
    point_nodes: list


def _require_gmsh():
    try:
        import gmsh
    except ImportError as err:                  # pragma: no cover - depends on the machine
        raise ImportError("meshing a site needs gmsh: pip install gmsh.  On a server with no "
                          "display, gmsh also needs the system libraries libglu1-mesa, "
                          "libxcursor1, libxinerama1 and libxft2") from err
    except OSError as err:                      # pragma: no cover - missing shared libraries
        raise ImportError(f"gmsh is installed but will not load ({err}).  On a server with no "
                          "display install libglu1-mesa, libxcursor1, libxinerama1 and libxft2, "
                          "or gmsh's build without X from https://gmsh.info/python-packages-dev-nox") from err
    return gmsh


def mesh_site(profile: SoilProfile, x: tuple[float, float], y: tuple[float, float],
              excavations: list[Excavation] = (), mesh_size: float = 2.0,
              sample_spacing: float | None = None, verbose: bool = False,
              walls: list[SiteWall] = (), points=(), fills=()) -> SiteMesh:
    """Mesh a site; ``mesh.region`` is the soil index of every element.

    ``walls`` become surfaces of element faces - through the soil, or
    embedded in it where a wall stops short of the base - and every point in
    ``points`` becomes a node, so an anchor can end exactly where it should.
    ``fills`` are meshed above the ground; their elements have region
    ``n_soils + i`` for fill ``i`` and their lifts are groups.
    """
    gmsh = _require_gmsh()
    (x0, x1), (y0, y1) = x, y
    if not (x1 > x0 and y1 > y0):
        raise ValueError("the plan extent must have positive size")
    for exc in excavations:
        p = np.asarray(exc.polygon, float)
        if p[:, 0].min() < x0 or p[:, 0].max() > x1 or p[:, 1].min() < y0 or p[:, 1].max() > y1:
            raise ValueError(f"excavation {exc.name!r} reaches outside the model")

    size = max(x1 - x0, y1 - y0)
    margin = 0.05 * size
    spacing = sample_spacing or max(mesh_size, size / 40)
    gx = np.linspace(x0 - margin, x1 + margin, max(2, int(np.ceil((x1 - x0 + 2 * margin) / spacing)) + 1))
    gy = np.linspace(y0 - margin, y1 + margin, max(2, int(np.ceil((y1 - y0 + 2 * margin) / spacing)) + 1))
    X, Y = np.meshgrid(gx, gy, indexing="xy")                  # rows along x, one per y
    plan = np.column_stack([X.ravel(), Y.ravel()])
    tops = profile.tops(plan).reshape(len(gy), len(gx), profile.n_soils)
    z_high = float(tops[..., 0].max()) + max(1.0, 0.05 * size)

    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("site")
        occ = gmsh.model.occ

        def loft(z: np.ndarray) -> int:
            """A surface through the sampled levels z (ny, nx)."""
            wires = []
            for j in range(len(gy)):
                pts = [occ.addPoint(gx[i], gy[j], z[j, i]) for i in range(len(gx))]
                curve = occ.addSpline(pts) if len(pts) > 2 else occ.addLine(*pts)
                wires.append(occ.addWire([curve]))
            out = occ.addThruSections(wires, makeSolid=False, makeRuled=len(gy) == 2)
            return out[0][1]

        box = occ.addBox(x0, y0, profile.bottom, x1 - x0, y1 - y0, z_high - profile.bottom)
        ground = loft(tops[..., 0])
        air = occ.extrude([(2, ground)], 0, 0, z_high - float(tops[..., 0].min()) + 1.0)
        air_volumes = [e for e in air if e[0] == 3]
        domain, _ = occ.cut([(3, box)], air_volumes)

        # fills: each lift a prism over its polygon, less the ground under it
        for fill in fills:
            poly = np.asarray(fill.polygon, float)
            if poly[:, 0].min() < x0 - 1e-9 or poly[:, 0].max() > x1 + 1e-9 or \
                    poly[:, 1].min() < y0 - 1e-9 or poly[:, 1].max() > y1 + 1e-9:
                raise ValueError(f"fill {fill.name!r} reaches outside the model")
            for k, level in enumerate(fill.levels):
                lower = float(tops[..., 0].min()) - 1.0 if k == 0 else fill.levels[k - 1]
                pts = [occ.addPoint(px, py, lower) for px, py in poly]
                lines = [occ.addLine(a, b) for a, b in zip(pts, pts[1:] + pts[:1])]
                face = occ.addPlaneSurface([occ.addCurveLoop(lines)])
                prism = [e for e in occ.extrude([(2, face)], 0, 0, level - lower) if e[0] == 3]
                above, _ = occ.cut(prism, domain, removeObject=True, removeTool=False)
                if not above:
                    raise ValueError(f"fill {fill.name!r}: lift {k + 1} lies below the ground")
                domain = list(domain) + list(above)

        tools = []
        for k in range(1, profile.n_soils):
            z = tops[..., k]
            # a soil boundary that never separates anything (a soil missing
            # everywhere, or one sitting on the base) is not a surface at all
            if np.all(z <= profile.bottom + 1e-9) or np.allclose(z, tops[..., k - 1]):
                continue
            tools.append((2, loft(z)))

        for exc in excavations:
            poly = np.asarray(exc.polygon, float)
            for k, level in enumerate(exc.levels):
                upper = z_high if k == 0 else exc.levels[k - 1]
                pts = [occ.addPoint(px, py, level) for px, py in poly]
                lines = [occ.addLine(a, b) for a, b in zip(pts, pts[1:] + pts[:1])]
                face = occ.addPlaneSurface([occ.addCurveLoop(lines)])
                prism = [e for e in occ.extrude([(2, face)], 0, 0, upper - level) if e[0] == 3]
                clipped, _ = occ.intersect(prism, domain, removeObject=True, removeTool=False)
                tools.extend(clipped)

        # walls: a vertical rectangle per segment, clipped to the ground
        wall_tools = {}
        for wall in walls:
            path = np.asarray(wall.path, float)
            top = z_high if wall.top is None else wall.top
            wall_tools[wall.name] = []
            for (xa, ya), (xb, yb) in zip(path[:-1], path[1:]):
                corners = [occ.addPoint(xa, ya, wall.toe), occ.addPoint(xb, yb, wall.toe),
                           occ.addPoint(xb, yb, top), occ.addPoint(xa, ya, top)]
                lines = [occ.addLine(a, b) for a, b in zip(corners, corners[1:] + corners[:1])]
                face = occ.addPlaneSurface([occ.addCurveLoop(lines)])
                clipped, _ = occ.intersect([(2, face)], domain, removeObject=True, removeTool=False)
                if not clipped:
                    raise ValueError(f"wall {wall.name!r} has a segment wholly outside the ground")
                wall_tools[wall.name] += list(range(len(tools), len(tools) + len(clipped)))
                tools.extend(clipped)
        point_tools = []
        for px, py, pz in points:
            point_tools.append(len(tools))
            tools.append((0, occ.addPoint(px, py, pz)))

        # fragment even with nothing to cut: fills must share their faces with the ground
        out_map = occ.fragment(domain, tools)[1] if tools or len(domain) > 1 else []
        occ.synchronize()
        offset = len(domain)
        wall_surfaces = {name: sorted({t for i in idx for d, t in out_map[offset + i] if d == 2})
                         for name, idx in wall_tools.items()}
        point_entities = []
        for i in point_tools:
            found = [t for d, t in out_map[offset + i] if d == 0]
            if not found:
                raise ValueError(f"point {points[len(point_entities)]} lies outside the ground")
            point_entities.append(found[0])

        volumes = gmsh.model.getEntities(3)
        keep = {abs(t) for _, t in gmsh.model.getBoundary(volumes, combined=False, oriented=False)}
        # a wall that stops short of the base is embedded in a volume rather
        # than bounding one: it must not be taken for a leftover
        keep |= {t for tags in wall_surfaces.values() for t in tags}
        stray = [(2, t) for _, t in gmsh.model.getEntities(2) if t not in keep]
        if stray:
            occ.remove(stray, recursive=True)
            occ.synchronize()

        # sizes: the global size everywhere, finer round each excavation
        fields = []
        for fill in fills:
            if fill.mesh_size:
                poly = np.asarray(fill.polygon, float)
                f = gmsh.model.mesh.field.add("Box")
                gmsh.model.mesh.field.setNumber(f, "VIn", fill.mesh_size)
                gmsh.model.mesh.field.setNumber(f, "VOut", mesh_size)
                gmsh.model.mesh.field.setNumber(f, "XMin", poly[:, 0].min())
                gmsh.model.mesh.field.setNumber(f, "XMax", poly[:, 0].max())
                gmsh.model.mesh.field.setNumber(f, "YMin", poly[:, 1].min())
                gmsh.model.mesh.field.setNumber(f, "YMax", poly[:, 1].max())
                gmsh.model.mesh.field.setNumber(f, "ZMin", profile.bottom)
                gmsh.model.mesh.field.setNumber(f, "ZMax", max(fill.levels))
                gmsh.model.mesh.field.setNumber(f, "Thickness", mesh_size)
                fields.append(f)
        for exc in excavations:
            poly = np.asarray(exc.polygon, float)
            depth = float(tops[..., 0].max() - exc.levels[-1])
            f = gmsh.model.mesh.field.add("Box")
            gmsh.model.mesh.field.setNumber(f, "VIn", exc.mesh_size or 0.5 * mesh_size)
            gmsh.model.mesh.field.setNumber(f, "VOut", mesh_size)
            gmsh.model.mesh.field.setNumber(f, "XMin", poly[:, 0].min() - depth)
            gmsh.model.mesh.field.setNumber(f, "XMax", poly[:, 0].max() + depth)
            gmsh.model.mesh.field.setNumber(f, "YMin", poly[:, 1].min() - depth)
            gmsh.model.mesh.field.setNumber(f, "YMax", poly[:, 1].max() + depth)
            gmsh.model.mesh.field.setNumber(f, "ZMin", exc.levels[-1] - depth)
            gmsh.model.mesh.field.setNumber(f, "ZMax", z_high)
            gmsh.model.mesh.field.setNumber(f, "Thickness", 2 * depth)
            fields.append(f)
        # Soil surfaces that pass close to a pit corner, or a layer pinching
        # out against the model's side, leave edges millimetres long.  The
        # mesh must resolve them, and unless the element size grows away from
        # them gradually the elements beside them are slivers.  Each short
        # edge therefore gets a size field starting at its own length and
        # growing linearly to the global size over three element lengths.
        for _, curve in gmsh.model.getEntities(1):
            length = occ.getMass(1, curve)
            if length < 0.25 * mesh_size:
                dist = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(dist, "CurvesList", [curve])
                gmsh.model.mesh.field.setNumber(dist, "Sampling", 10)
                f = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(f, "InField", dist)
                gmsh.model.mesh.field.setNumber(f, "SizeMin", max(length, 1e-3 * mesh_size))
                gmsh.model.mesh.field.setNumber(f, "SizeMax", mesh_size)
                gmsh.model.mesh.field.setNumber(f, "DistMin", 0.0)
                gmsh.model.mesh.field.setNumber(f, "DistMax", 3.0 * mesh_size)
                fields.append(f)
        if fields:
            fmin = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(fmin, "FieldsList", fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(fmin)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        # A point inside a volume should be embedded in it by the fragment,
        # but OpenCASCADE's inside test can fail near a lofted surface and the
        # point is then silently left out.  Find the volume from a first mesh
        # instead - our own barycentric test - and embed it explicitly.
        attached = set()
        for dim in (1, 2, 3):
            for _, tag in gmsh.model.getEntities(dim):
                attached |= {t for d, t in gmsh.model.mesh.getEmbedded(dim, tag) if d == 0}
                attached |= {abs(t) for d, t in gmsh.model.getBoundary([(dim, tag)], recursive=True) if d == 0}
        lost = [(i, tag) for i, tag in enumerate(point_entities) if tag not in attached]
        # HXT: on thin slabs of soil (a lift floor half a metre above a layer
        # boundary) it avoids the near-flat elements the default Delaunay
        # mesher leaves there - worst radius ratio 0.30 against 0.01.  One
        # thread, so the same site always gives the same mesh.
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.model.mesh.generate(3)
        if lost:
            node_tags, coords, _ = gmsh.model.mesh.getNodes()
            where = dict(zip(node_tags.astype(np.int64), coords.reshape(-1, 3)))
            for i, point in lost:
                q = np.asarray(points[i], float)
                host = None
                for _, vol in gmsh.model.getEntities(3):
                    types, _, conn = gmsh.model.mesh.getElements(3, vol)
                    tets = np.asarray(conn[0], dtype=np.int64).reshape(-1, 4)
                    P = np.stack([np.array([where[t] for t in col]) for col in tets.T], axis=1)
                    lam = np.linalg.solve(np.transpose(P[:, 1:] - P[:, :1], (0, 2, 1)),
                                          (q - P[:, 0])[:, :, None])[:, :, 0]
                    if np.any(np.all(np.column_stack([1 - lam.sum(axis=1), lam]) >= -1e-9, axis=1)):
                        host = vol
                        break
                if host is None:
                    raise ValueError(f"point {tuple(points[i])} lies outside the ground")
                gmsh.model.mesh.embed(0, [point], 3, host)
            gmsh.model.mesh.clear()
            gmsh.model.mesh.generate(3)

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = coords.reshape(-1, 3)
        wall_triangles = {}
        for name, tags in wall_surfaces.items():
            tris = []
            for tag in tags:
                types, _, conn = gmsh.model.mesh.getElements(2, tag)
                for etype, c in zip(types, conn):
                    if etype == 2:
                        tris.append(np.asarray(c, dtype=np.int64).reshape(-1, 3))
            wall_triangles[name] = np.concatenate(tris) if tris else np.zeros((0, 3), np.int64)
        point_tags = [int(gmsh.model.mesh.getNodes(0, tag)[0][0]) for tag in point_entities]
        tets, volume_of = [], []
        for _, tag in gmsh.model.getEntities(3):
            types, _, conn = gmsh.model.mesh.getElements(3, tag)
            for etype, c in zip(types, conn):
                if etype != 4:
                    raise RuntimeError(f"gmsh produced element type {etype}, expected linear tetrahedra")
                tets.append(np.asarray(c, dtype=np.int64).reshape(-1, 4))
                volume_of.append(np.full(len(tets[-1]), tag))
    finally:
        gmsh.finalize()

    tets = np.concatenate(tets)
    volume_of = np.concatenate(volume_of)
    # compact numbering over the nodes the tetrahedra use
    index = np.full(int(node_tags.max()) + 1, -1, dtype=np.int64)
    index[node_tags.astype(np.int64)] = np.arange(len(node_tags))
    tets = index[tets]
    used, tets = np.unique(tets, return_inverse=True)
    tets = tets.reshape(-1, 4)
    nodes = coords[used]
    renumber = np.full(len(node_tags), -1, dtype=np.int64)
    renumber[used] = np.arange(len(used))

    p = nodes[tets]
    vol = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
    flip = vol < 0
    tets[flip, 1], tets[flip, 2] = tets[flip, 2].copy(), tets[flip, 1].copy()
    mesh = quadratic_from_linear(nodes, tets)

    # label every gmsh volume by the majority of its elements' centroids
    centroids = mesh.centroids()
    soil = profile.soil_of(centroids)
    region = np.empty(mesh.n_elements, dtype=np.int64)
    groups = {name: np.zeros(mesh.n_elements, bool)
              for item in list(excavations) + list(fills) for name in item.lift_names}
    lifts = [(exc, exc.lift_of(centroids)) for exc in excavations]
    ground = profile.ground(centroids)
    fill_lifts = [(i, f, f.lift_of(centroids, ground)) for i, f in enumerate(fills)]
    for tag in np.unique(volume_of):
        members = volume_of == tag
        region[members] = np.bincount(soil[members]).argmax()
        for i, fill, lift in fill_lifts:
            votes = np.bincount(lift[members] + 1, minlength=len(fill.levels) + 1)
            k = int(votes.argmax()) - 1
            if k >= 0:
                region[members] = profile.n_soils + i
                groups[fill.lift_names[k]][members] = True
        for exc, lift in lifts:
            votes = np.bincount(lift[members] + 1, minlength=len(exc.levels) + 1)
            k = int(votes.argmax()) - 1
            if k >= 0:
                groups[exc.lift_names[k]][members] = True
    mesh.region = region
    wall_faces = {name: mesh.faces_from_corners(renumber[index[tris]]) for name, tris in wall_triangles.items()}
    point_nodes = [int(renumber[index[t]]) for t in point_tags]
    if any(n < 0 for n in point_nodes):
        raise RuntimeError("a point was not meshed into the volume")
    if any(np.linalg.norm(nodes[n] - np.asarray(q, float)) > 1e-6 * size for n, q in zip(point_nodes, points)):
        raise RuntimeError("a point's node is not where the point is")
    return SiteMesh(mesh, groups, wall_faces, point_nodes)
