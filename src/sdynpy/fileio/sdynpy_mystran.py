# -*- coding: utf-8 -*-
"""
Neal Gordon  
2026-10-22  

Readers for MYSTRAN input and output files.

more info on the solver can be found at  
https://mystran.com/

Reads the bulk data (.dat), the printed results (.F06), and the binary
OUTPUT4 matrices (.OU4) and converts them into SDynPy geometry and shape
objects.  Only numpy and scipy are required.

Coordinate systems are currently ignored: all nodes are placed in a single
global Cartesian coordinate system and the CP/CD fields of the GRID cards are
parsed but not used.

"""

import os
import re
import warnings

import numpy as np
from scipy.io import FortranEOFError, FortranFile

from ..core.sdynpy_coordinate import coordinate_array
from ..core.sdynpy_geometry import (Geometry, coordinate_system_array,
                                    element_array, node_array)
from ..core.sdynpy_shape import shape_array

_FLOAT = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'


# ---------------------------------------------------------------------------
# F06
# ---------------------------------------------------------------------------

def read_f06(path):
    """
    Read eigenvalues and eigenvectors from a MYSTRAN F06 file.

    Parameters
    ----------
    path : str
        Path to the F06 file.

    Returns
    -------
    dict
        Dictionary with keys

        - ``eigenvalue`` : (nmodes,) eigenvalues (rad/s)^2
        - ``radians`` : (nmodes,) natural frequencies in rad/s
        - ``frequency`` : (nmodes,) natural frequencies in Hz
        - ``generalized_mass`` : (nmodes,)
        - ``generalized_stiffness`` : (nmodes,)
        - ``grid_id`` : (ngrids,) grid ids in the order printed
        - ``shape`` : (nmodes, ngrids, 6) eigenvectors (T1-T3, R1-R3)
    """
    row_eig = re.compile(r'^\s*(\d+)\s+(\d+)\s+((?:' + _FLOAT + r'\s*){5})\s*$')
    row_vec = re.compile(r'^\s*(\d+)\s+(\d+)\s+((?:' + _FLOAT + r'\s*){6})\s*$')
    eig_rows = []
    vectors = {}
    grid_ids = {}
    in_eig_table = False
    current = None
    with open(path, 'r', errors='replace') as f:
        for line in f:
            if 'R E A L   E I G E N V A L U E S' in line:
                in_eig_table = True
                continue
            m = re.match(r'^\s*OUTPUT FOR EIGENVECTOR\s+(\d+)', line)
            if m:
                in_eig_table = False
                current = int(m.group(1))
                vectors[current] = []
                grid_ids[current] = []
                continue
            if in_eig_table:
                m = row_eig.match(line)
                if m:
                    eig_rows.append([int(m.group(1))] + [float(v) for v in m.group(3).split()])
                elif eig_rows and line.strip() == '':
                    in_eig_table = False
                continue
            if current is not None:
                if '------' in line:
                    current = None
                    continue
                m = row_vec.match(line)
                if m:
                    grid_ids[current].append(int(m.group(1)))
                    vectors[current].append([float(v) for v in m.group(3).split()])
    if not eig_rows:
        raise ValueError('No eigenvalue table found in {:}'.format(path))
    eig = np.array(eig_rows)
    mode_numbers = eig[:, 0].astype(int)
    missing = [n for n in mode_numbers if n not in vectors]
    if missing:
        raise ValueError('No eigenvectors found in F06 for modes {:}'.format(missing))
    ids = np.array(grid_ids[mode_numbers[0]], dtype=int)
    shape = np.array([vectors[n] for n in mode_numbers])
    return {
        'eigenvalue': eig[:, 1],
        'radians': eig[:, 2],
        'frequency': eig[:, 3],
        'generalized_mass': eig[:, 4],
        'generalized_stiffness': eig[:, 5],
        'grid_id': ids,
        'shape': shape,
    }


# ---------------------------------------------------------------------------
# OU4
# ---------------------------------------------------------------------------

def read_ou4(path):
    """
    Read the matrices from a MYSTRAN OUTPUT4 binary (.OU4) file.

    The file is a sequence of Fortran unformatted records.  Each matrix has
    a 24 byte header (ncols, nrows, form, type, 8 character name) followed by
    one record per column holding (column, first row, number of 4 byte words)
    and the column values.

    Parameters
    ----------
    path : str
        Path to the OU4 file.

    Returns
    -------
    dict
        Matrix name (as stored, at most 8 characters, e.g. ``EIGEN_VA``)
        mapped to a dense 2D array.
    """
    matrices = {}
    with FortranFile(path, 'r') as f:
        while True:
            try:
                rec = f.read_record(np.uint8)
            except (EOFError, FortranEOFError):
                break
            if rec.size != 24:
                continue
            ncols, nrows, form, mtype = (int(v) for v in np.frombuffer(rec[:16], dtype=np.int32))
            name = rec[16:24].tobytes().decode(errors='ignore').strip()
            if ncols < 1 or nrows < 1 or name == '' or mtype not in (1, 2):
                continue
            dtype = np.float32 if mtype == 1 else np.float64
            mat = np.zeros((nrows, ncols), dtype=dtype)
            for _ in range(ncols):
                col_rec = f.read_record(np.uint8)
                col, first_row, _ = (int(v) for v in np.frombuffer(col_rec[:12], dtype=np.int32))
                values = np.frombuffer(col_rec[12:], dtype=dtype)
                mat[first_row - 1:first_row - 1 + values.size, col - 1] = values
            matrices[name] = mat
    return matrices


# ---------------------------------------------------------------------------
# DAT
# ---------------------------------------------------------------------------

def _to_float(s):
    s = s.strip()
    if s == '':
        return 0.0
    try:
        return float(s)
    except ValueError:
        # Nastran shorthand exponent, e.g. 1.0-3 -> 1.0e-3
        return float(re.sub(r'^([-+]?[\d.]+)([-+]\d+)$', r'\1e\2', s.replace('D', 'E').replace('d', 'e')))


def _to_int(s):
    s = s.strip()
    return int(s) if s != '' else 0


def _split_fields(line):
    """Return (name, fields) for a bulk data line, padded to 8 data fields."""
    if ',' in line:
        parts = [p.strip() for p in line.split(',')]
        name, fields = parts[0], parts[1:]
    else:
        name = line[:8].strip()
        fields = [line[i:i + 8].strip() for i in range(8, 72, 8)]
    fields = fields[:8] + [''] * (8 - len(fields))
    return name, fields


def _bulk_cards(path):
    """Yield (name, fields) for each bulk data card with continuations merged."""
    in_bulk = False
    case_control = []
    card = None
    with open(path, 'r', errors='replace') as f:
        for raw in f:
            line = raw.rstrip('\r\n').split('$', 1)[0].rstrip()
            if not in_bulk:
                if line.strip().upper().startswith('BEGIN') and 'BULK' in line.upper():
                    in_bulk = True
                else:
                    case_control.append(line)
                continue
            if line.strip() == '':
                continue
            if line.strip().upper().startswith('ENDDATA'):
                break
            if line[0] in ' +*,':
                if card is None:
                    continue
                if ',' in line:
                    cont = [p.strip() for p in line.split(',')[1:]]
                else:
                    cont = [line[i:i + 8].strip() for i in range(8, 72, 8)]
                card[1].extend(cont[:8] + [''] * (8 - len(cont)))
                continue
            if card is not None:
                yield card
            name, fields = _split_fields(line)
            if name.endswith('*'):
                raise NotImplementedError('Large field format cards are not supported: ' + name)
            card = (name.upper(), fields)
    if card is not None:
        yield card
    _bulk_cards.case_control = case_control


def read_dat(path):
    """
    Read the grids, CBAR elements and single point constraints from a MYSTRAN
    bulk data file.

    Parameters
    ----------
    path : str
        Path to the .dat file.

    Returns
    -------
    dict
        Dictionary with keys

        - ``grid_id`` : (ngrids,) grid ids in sorted order
        - ``grid_coordinate`` : (ngrids, 3) positions
        - ``grid_cp`` : (ngrids,) position coordinate system ids (unused)
        - ``grid_cd`` : (ngrids,) displacement coordinate system ids (unused)
        - ``cbar_id``, ``cbar_connectivity`` : CBAR ids and (n, 2) grid ids
        - ``constraints`` : dict mapping grid id to a set of constrained
          components (1-6), from the SPC1 set selected in the case control
          (all SPC1 sets if none is selected) and the GRID PS field
    """
    grids = {}
    cbars = {}
    spc1 = []
    ps = {}
    for name, f in _bulk_cards(path):
        if name == 'GRID':
            gid = _to_int(f[0])
            grids[gid] = (_to_int(f[1]), [_to_float(v) for v in f[2:5]], _to_int(f[5]))
            if f[6]:
                ps[gid] = {int(c) for c in f[6] if c in '123456'}
        elif name == 'CBAR':
            cbars[_to_int(f[0])] = (_to_int(f[2]), _to_int(f[3]))
        elif name == 'SPC1':
            spc1.append((_to_int(f[0]), f[1], [v for v in f[2:] if v != '']))
    case_control = getattr(_bulk_cards, 'case_control', [])
    selected = None
    for line in case_control:
        m = re.match(r'^\s*SPC\s*=\s*(\d+)', line, re.IGNORECASE)
        if m:
            selected = int(m.group(1))
    grid_ids = np.array(sorted(grids), dtype=int)
    constraints = {}
    for sid, comps, nodes in spc1:
        if selected is not None and sid != selected:
            continue
        if 'THRU' in [n.upper() for n in nodes]:
            i = [n.upper() for n in nodes].index('THRU')
            lo, hi = int(nodes[i - 1]), int(nodes[i + 1])
            nodes = nodes[:i - 1] + [str(g) for g in grid_ids if lo <= g <= hi] + nodes[i + 2:]
        for n in nodes:
            constraints.setdefault(int(n), set()).update(int(c) for c in comps if c in '123456')
    for gid, comps in ps.items():
        constraints.setdefault(gid, set()).update(comps)
    if any(cp != 0 or cd != 0 for cp, _, cd in grids.values()):
        warnings.warn('GRID coordinate systems (CP/CD) are not supported yet and are ignored.')
    cbar_ids = np.array(sorted(cbars), dtype=int)
    return {
        'grid_id': grid_ids,
        'grid_coordinate': np.array([grids[g][1] for g in grid_ids]).reshape(-1, 3),
        'grid_cp': np.array([grids[g][0] for g in grid_ids], dtype=int),
        'grid_cd': np.array([grids[g][2] for g in grid_ids], dtype=int),
        'cbar_id': cbar_ids,
        'cbar_connectivity': np.array([cbars[e] for e in cbar_ids], dtype=int).reshape(-1, 2),
        'constraints': constraints,
    }


def free_dofs(model):
    """
    Return the unconstrained (L-set) degrees of freedom of a model.

    Parameters
    ----------
    model : dict
        Output of :func:`read_dat`

    Returns
    -------
    nodes, directions : np.ndarray
        Grid ids and directions (1-6) in grid-major order, which is the row
        order of the L-set matrices in the OU4 file.
    """
    nodes = []
    directions = []
    for g in model['grid_id']:
        fixed = model['constraints'].get(int(g), set())
        for d in range(1, 7):
            if d not in fixed:
                nodes.append(g)
                directions.append(d)
    return np.array(nodes, dtype=int), np.array(directions, dtype=int)


# ---------------------------------------------------------------------------
# SDynPy objects
# ---------------------------------------------------------------------------

def geometry_from_model(model):
    """
    Create an SDynPy Geometry from the output of :func:`read_dat`.

    All nodes are placed in a single global Cartesian coordinate system
    with id 1.  CBAR elements become linear beam elements (type 21).
    """
    nodes = node_array(model['grid_id'], model['grid_coordinate'], 1, 1, 1)
    cs = coordinate_system_array(1, 'Global')
    if model['cbar_id'].size:
        elements = element_array(model['cbar_id'], 21, 1, model['cbar_connectivity'])
        return Geometry(nodes, cs, element=elements)
    return Geometry(nodes, cs)


def _select_dofs(grid_id, shape, nodes, directions):
    """Pick (nmodes, ndof) shape coefficients for the node/direction pairs."""
    index = {int(g): i for i, g in enumerate(grid_id)}
    cols = [index[int(n)] * 6 + int(d) - 1 for n, d in zip(nodes, directions)]
    return shape.reshape(shape.shape[0], -1)[:, cols]


def shapes_from_f06(f06, model=None, damping=0.0):
    """
    Create a ShapeArray from the output of :func:`read_f06`.

    Parameters
    ----------
    f06 : dict
        Output of :func:`read_f06`
    model : dict, optional
        Output of :func:`read_dat`.  If given, only unconstrained degrees of
        freedom are kept, otherwise all six DOFs of every grid are kept.
    damping : float or np.ndarray, optional
        Fraction of critical damping assigned to the shapes.
    """
    if model is None:
        nodes = np.repeat(f06['grid_id'], 6)
        directions = np.tile(np.arange(1, 7), f06['grid_id'].size)
    else:
        nodes, directions = free_dofs(model)
    matrix = _select_dofs(f06['grid_id'], f06['shape'], nodes, directions)
    return shape_array(coordinate_array(nodes, directions), matrix,
                       f06['frequency'], damping, f06['generalized_mass'])


def shapes_from_ou4(ou4, model, damping=0.0):
    """
    Create a ShapeArray from the matrices returned by :func:`read_ou4`.

    The EIGEN_VEC matrix is assumed to be in the G-set (6 DOFs per grid) if
    it has 6 times as many rows as grids, and in the L-set otherwise.  The
    OU4 file does not contain generalized masses, so a modal mass of 1 is
    assigned (MYSTRAN mass normalizes the vectors by default).
    """
    vec = ou4['EIGEN_VE'].T
    val = np.asarray(ou4['EIGEN_VA']).reshape(-1)
    frequency = np.sqrt(np.abs(val)) / (2 * np.pi)
    if vec.shape[1] == 6 * model['grid_id'].size:
        grid_nodes = np.repeat(model['grid_id'], 6)
        grid_dirs = np.tile(np.arange(1, 7), model['grid_id'].size)
        keep = [i for i, (n, d) in enumerate(zip(grid_nodes, grid_dirs))
                if d not in model['constraints'].get(int(n), set())]
        nodes, directions = grid_nodes[keep], grid_dirs[keep]
        vec = vec[:, keep]
    else:
        nodes, directions = free_dofs(model)
        if vec.shape[1] != nodes.size:
            raise ValueError('EIGEN_VEC has {:} DOFs but the model has {:} free DOFs'.format(
                vec.shape[1], nodes.size))
    return shape_array(coordinate_array(nodes, directions), vec, frequency, damping, 1.0)


def read_mystran(basename, source='f06', damping=0.0):
    """
    Read a MYSTRAN model and its modal results.

    Parameters
    ----------
    basename : str
        Path to the model without extension (or to the .dat file).  The
        .dat, .F06 and .OU4 files are expected next to each other.
    source : str, optional
        'f06' or 'ou4', which file to read the modes from.  The default is
        'f06'.
    damping : float, optional
        Fraction of critical damping assigned to every mode.

    Returns
    -------
    geometry : Geometry
    shapes : ShapeArray
    """
    base, ext = os.path.splitext(basename)
    if ext.lower() not in ('.dat', '.f06', '.ou4'):
        base = basename

    def find(ext):
        for candidate in (ext, ext.lower(), ext.upper()):
            if os.path.exists(base + candidate):
                return base + candidate
        raise FileNotFoundError(base + ext)

    model = read_dat(find('.dat'))
    geometry = geometry_from_model(model)
    if source.lower() == 'f06':
        shapes = shapes_from_f06(read_f06(find('.F06')), model, damping)
    elif source.lower() == 'ou4':
        shapes = shapes_from_ou4(read_ou4(find('.OU4')), model, damping)
    else:
        raise ValueError("source must be 'f06' or 'ou4'")
    return geometry, shapes
