"""VTK unstructured grid (.vtu) output, for viewing results in ParaView.

Arrays are written in VTK's inline binary encoding (base64 with a 64-bit
length header), which ParaView reads directly and which is several times
smaller than ASCII.  Elements go out as VTK quadratic tetrahedra (cell type
24), whose node order the elements already follow.
"""

from __future__ import annotations

import base64
import os

import numpy as np

VTK_QUADRATIC_TETRA = 24
VTK_QUADRATIC_TRIANGLE = 22

_TYPES = {
    np.dtype(np.float64): "Float64",
    np.dtype(np.float32): "Float32",
    np.dtype(np.int64): "Int64",
    np.dtype(np.int32): "Int32",
    np.dtype(np.uint8): "UInt8",
}

#: names of the six Voigt components, in storage order
VOIGT = ("xx", "yy", "zz", "xy", "yz", "zx")


def _array(name: str, data: np.ndarray, indent: str) -> str:
    data = np.ascontiguousarray(data)
    if data.dtype not in _TYPES:
        data = data.astype(np.float64)
    ncomp = 1 if data.ndim == 1 else data.shape[1]
    raw = data.tobytes()
    blob = base64.b64encode(np.uint64(len(raw)).tobytes() + raw).decode("ascii")
    extra = f' NumberOfComponents="{ncomp}"' if ncomp > 1 else ""
    if data.ndim == 2 and ncomp == 6:
        extra += "".join(f' ComponentName{i}="{c}"' for i, c in enumerate(VOIGT))
    return (f'{indent}<DataArray type="{_TYPES[data.dtype]}" Name="{name}"{extra} '
            f'format="binary">{blob}</DataArray>\n')


def write_vtu(path: str | os.PathLike, nodes: np.ndarray, elements: np.ndarray,
              point_data: dict | None = None, cell_data: dict | None = None) -> str:
    """Write a mesh of 10-node tetrahedra (or 6-node triangles) and its fields to ``path``."""
    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)
    ne, npe = elements.shape if elements.ndim == 2 else (0, 10)
    cell_type = {10: VTK_QUADRATIC_TETRA, 6: VTK_QUADRATIC_TRIANGLE}[npe]
    parts = [
        '<?xml version="1.0"?>\n',
        '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" '
        'header_type="UInt64">\n',
        "  <UnstructuredGrid>\n",
        f'    <Piece NumberOfPoints="{len(nodes)}" NumberOfCells="{ne}">\n',
    ]
    for tag, data in (("PointData", point_data), ("CellData", cell_data)):
        if data:
            parts.append(f"      <{tag}>\n")
            parts += [_array(k, np.asarray(v), "        ") for k, v in data.items()]
            parts.append(f"      </{tag}>\n")
    parts += [
        "      <Points>\n",
        _array("Points", nodes, "        "),
        "      </Points>\n",
        "      <Cells>\n",
        _array("connectivity", elements.ravel(), "        "),
        _array("offsets", npe * np.arange(1, ne + 1, dtype=np.int64), "        "),
        _array("types", np.full(ne, cell_type, dtype=np.uint8), "        "),
        "      </Cells>\n",
        "    </Piece>\n",
        "  </UnstructuredGrid>\n",
        "</VTKFile>\n",
    ]
    with open(path, "w", encoding="ascii") as fh:
        fh.writelines(parts)
    return str(path)


def read_vtu_array(text: str, name: str) -> np.ndarray:
    """Decode one binary array from the text of a file written by :func:`write_vtu`."""
    import xml.etree.ElementTree as ET

    for node in ET.fromstring(text).iter("DataArray"):
        if node.get("Name") == name:
            dtype = {v: k for k, v in _TYPES.items()}[node.get("type")]
            raw = base64.b64decode(node.text.strip())
            n = int(np.frombuffer(raw[:8], dtype=np.uint64)[0])
            data = np.frombuffer(raw[8:8 + n], dtype=dtype)
            ncomp = int(node.get("NumberOfComponents", 1))
            return data.reshape(-1, ncomp) if ncomp > 1 else data
    raise KeyError(name)


def write_result(path: str | os.PathLike, result) -> str:
    """Write a :class:`~lythos3d.core.analysis.LinearResult` for ParaView."""
    mesh = result.mesh
    return write_vtu(
        path, mesh.nodes, mesh.elements,
        point_data={
            "displacement": result.displacement,
            "stress": result.nodal_stress,
        },
        cell_data={
            "region": mesh.region,
            "mean_stress": result.stress.reshape(mesh.n_elements, 4, 6)[:, :, :3].mean(axis=(1, 2)),
        },
    )


def write_stage(path: str | os.PathLike, problem, result) -> str:
    """Write one :class:`~lythos3d.core.solver.StageResult` for ParaView.

    Only the elements present at that stage are written, so excavated ground
    is simply absent.  ``plastic_strain`` (equivalent plastic strain, mean per
    element) is what shows a failure mechanism: after a strength reduction it
    traces the slip surface.  ``stress`` is effective; with groundwater
    ``pore_pressure`` and ``total_stress`` are written as well.
    """
    mesh = problem.mesh
    ce = problem.continuum
    ngp = ce.n_gauss
    active = np.asarray(result.active, dtype=bool)
    nodal_stress = ce.nodal_average(np.where(np.repeat(active, ngp)[:, None], result.state.stress, 0.0),
                                    mesh.n_nodes)
    # average only over the active elements that touch each node
    count = np.zeros(mesh.n_nodes)
    np.add.at(count, mesh.elements[active].ravel(), 1.0)
    total = np.zeros(mesh.n_nodes)
    np.add.at(total, mesh.elements.ravel(), 1.0)
    nodal_stress *= (total / np.maximum(count, 1.0))[:, None]

    def per_element(values):
        return values.reshape(mesh.n_elements, ngp).mean(axis=1)[active]

    point_data = {"displacement": result.displacement, "stress": nodal_stress}
    pore = getattr(result, "pore_pressure", None)
    if pore is not None and np.any(pore):
        # the soil's stress is effective; total stress is sigma' - p m
        nodal_p = ce.nodal_average(np.where(np.repeat(active, ngp), pore, 0.0)[:, None], mesh.n_nodes)[:, 0]
        nodal_p *= total / np.maximum(count, 1.0)
        point_data["pore_pressure"] = nodal_p
        point_data["total_stress"] = nodal_stress - nodal_p[:, None] * np.array([1, 1, 1, 0, 0, 0], float)

    return write_vtu(
        path, mesh.nodes, mesh.elements[active],
        point_data=point_data,
        cell_data={
            "region": mesh.region[active],
            "plastic_strain": per_element(result.state.eps_p_eq),
            "plastic_fraction": per_element(result.state.yielding.astype(float)),
        },
    )


def write_plates(path: str | os.PathLike, problem, result) -> str | None:
    """Write the plates installed at a stage, with their forces per element, for ParaView.

    Moments and forces are element means in each element's own axes; the
    local axis most nearly vertical is reported as ``M_vertical`` (the
    bending moment of a wall about a horizontal axis) and the other as
    ``M_horizontal``.  Returns None when no plate is installed.
    """
    faces, fields = [], {k: [] for k in ("M_vertical", "M_horizontal", "M_twist", "N_vertical",
                                         "N_horizontal", "plate")}
    for k, (plate, el) in enumerate(zip(problem.plates, problem.plate_elements)):
        if plate.name not in result.plate_forces:
            continue
        N, M, _ = result.plate_forces[plate.name]
        Nm, Mm = N.mean(axis=1), M.mean(axis=1)
        vertical_is_x = np.abs(el.R[:, 0, 2]) >= np.abs(el.R[:, 1, 2])
        faces.append(plate.faces)
        fields["M_vertical"].append(np.where(vertical_is_x, Mm[:, 0], Mm[:, 1]))
        fields["M_horizontal"].append(np.where(vertical_is_x, Mm[:, 1], Mm[:, 0]))
        fields["M_twist"].append(Mm[:, 2])
        fields["N_vertical"].append(np.where(vertical_is_x, Nm[:, 0], Nm[:, 1]))
        fields["N_horizontal"].append(np.where(vertical_is_x, Nm[:, 1], Nm[:, 0]))
        fields["plate"].append(np.full(len(plate.faces), k, dtype=np.int64))
    if not faces:
        return None
    return write_vtu(path, problem.mesh.nodes, np.concatenate(faces),
                     point_data={"displacement": result.displacement},
                     cell_data={k: np.concatenate(v) for k, v in fields.items()})
