"""
NHG Vehicle Routing Problem Solver
Question 3: Relaxed delivery day schedule — Van fleet, CW + overnight
  Phase 0 — Geographic clustering to reassign delivery days
  Phase 1 — CW / NN construction + 2-opt improvement
  Phase 2 — Overnight route merging
Outputs: Console summary + Q3_RelaxedDays_Report.xlsx
"""

import pandas as pd
import numpy as np
from itertools import combinations
from copy import deepcopy
from collections import defaultdict

# ─────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────
VAN_CAPACITY = 3200
UNLOAD_RATE  = 0.03
MIN_UNLOAD   = 30
SPEED        = 40
SPEED_MPM    = 40 / 60

MAX_DRIVE_HR  = 11
MAX_DUTY_HR   = 14
BREAK_HR      = 10
MAX_DRIVE_MIN = MAX_DRIVE_HR * 60
MAX_DUTY_MIN  = MAX_DUTY_HR  * 60
BREAK_MIN     = BREAK_HR     * 60

WINDOW_OPEN  = 8  * 60
WINDOW_CLOSE = 18 * 60

DAY_ORDER = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("Loading data...")
orders_raw = pd.read_excel('deliveries.xlsx', sheet_name='OrderTable')
locs       = pd.read_excel('deliveries.xlsx', sheet_name='LocationTable')
dist_raw   = pd.read_excel('distances.xlsx', header=1, index_col=1)

orders = orders_raw[orders_raw['ORDERID'] != 0].copy()
orders['CUBE'] = pd.to_numeric(orders['CUBE'], errors='coerce')

locs = locs.dropna(subset=['ZIP', 'ZIPID'])
locs['ZIP']   = locs['ZIP'].astype(int)
locs['ZIPID'] = locs['ZIPID'].astype(int)
zip_to_id = dict(zip(locs['ZIP'], locs['ZIPID']))

# Lat/Lon lookups
locs_full = pd.read_excel('deliveries.xlsx', sheet_name='LocationTable')
locs_full = locs_full.dropna(subset=['ZIP', 'ZIPID'])
locs_full['ZIPID'] = locs_full['ZIPID'].astype(int)
zipid_to_x = dict(zip(locs_full['ZIPID'].astype(int), locs_full['X']))
zipid_to_y = dict(zip(locs_full['ZIPID'].astype(int), locs_full['Y']))
zipid_to_city = {}
zipid_to_state = {}
for _, row in locs_full.iterrows():
    zid = int(row['ZIPID'])
    if pd.notna(row.get('CITY', '')): zipid_to_city[zid] = str(row['CITY']).strip()
    if pd.notna(row.get('STATE', '')): zipid_to_state[zid] = str(row['STATE']).strip()

dist_matrix = {}
for row_id, row in dist_raw.iterrows():
    dist_matrix[int(row_id)] = {}
    for col_id in dist_raw.columns[1:]:
        dist_matrix[int(row_id)][int(col_id)] = int(row[col_id])

DEPOT_ID = 20

def get_dist(id1, id2):
    if id1 == id2: return 0
    return dist_matrix[id1][id2]

def unload_time(cube):
    return max(MIN_UNLOAD, UNLOAD_RATE * cube)

orders['ZIPID'] = orders['TOZIP'].map(lambda z: zip_to_id.get(int(z), None))


# ═══════════════════════════════════════════════════════════
#  PHASE 0: GEOGRAPHIC DAY REASSIGNMENT (balanced angular sweep)
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  PHASE 0: Reassigning delivery days via angular sweep")
print("=" * 60)

orig_day_counts = orders['DayOfWeek'].value_counts().to_dict()

# ── Step 1: Compute angle from depot for each store ──
depot_x = zipid_to_x[DEPOT_ID]
depot_y = zipid_to_y[DEPOT_ID]

store_orders = orders.groupby('TOZIP').agg(
    n_visits=('ORDERID', 'count'),
    order_ids=('ORDERID', list),
    zipid=('ZIPID', 'first')
).reset_index()

store_orders['angle'] = store_orders['zipid'].apply(
    lambda zid: np.arctan2(zipid_to_y.get(int(zid), depot_y) - depot_y,
                           zipid_to_x.get(int(zid), depot_x) - depot_x))

print(f"  {len(store_orders)} stores, {len(orders)} orders")

# ── Step 2: Sort stores by angle, divide into 5 balanced sectors ──
store_orders = store_orders.sort_values('angle').reset_index(drop=True)

# Walk through cumulative order count, cut sectors at ~52 order boundaries
target_per_day = len(orders) / 5  # ~52.2
sector_assignments = {}  # TOZIP -> sector index (0-4)
cumulative = 0
sector = 0
sector_cuts = []  # (sector_idx, n_stores, n_orders)
sector_store_count = 0
sector_order_count = 0

for i, srow in store_orders.iterrows():
    # If adding this store would push us past the target AND we still have sectors left
    if sector < 4 and cumulative + srow['n_visits'] > target_per_day * (sector + 1):
        # Check if it's closer to this sector or the next
        overshoot = (cumulative + srow['n_visits']) - target_per_day * (sector + 1)
        undershoot = target_per_day * (sector + 1) - cumulative
        if overshoot > undershoot and sector_order_count > 0:
            # Close current sector, start new one with this store
            sector_cuts.append((sector, sector_store_count, sector_order_count))
            sector += 1
            sector_store_count = 0
            sector_order_count = 0

    sector_assignments[srow['TOZIP']] = sector
    cumulative += srow['n_visits']
    sector_store_count += 1
    sector_order_count += srow['n_visits']

# Close the last sector
sector_cuts.append((sector, sector_store_count, sector_order_count))

# Map sectors to days
sector_to_day = {i: DAY_ORDER[i] for i in range(5)}

print(f"  Sector → Day mapping (angular sweep):")
for sec, n_st, n_ord in sector_cuts:
    print(f"    Sector {sec} → {sector_to_day[sec]:>3} : {n_st} stores, {n_ord} orders")

# ── Step 3: Assign ALL orders for each store to its sector's day ──
order_new_day = {}
day_counts = {d: 0 for d in DAY_ORDER}

for _, srow in store_orders.iterrows():
    sec = sector_assignments[srow['TOZIP']]
    day = sector_to_day[sec]
    for oid in srow['order_ids']:
        order_new_day[oid] = day
        day_counts[day] += 1

print(f"\n  New daily distribution:")
for d in DAY_ORDER:
    orig = orig_day_counts.get(d, 0)
    new = day_counts[d]
    print(f"    {d}: {orig:>3} → {new:>3} orders  (Δ {new - orig:+d})")

# ── Step 4: Apply new day assignments ──
orders['DayOfWeek'] = orders['ORDERID'].map(order_new_day)

day_orders = {}
for day in DAY_ORDER:
    day_df = orders[orders['DayOfWeek'] == day].copy()
    day_orders[day] = day_df.reset_index(drop=True)


# ─────────────────────────────────────────────
# ROUTE FEASIBILITY CHECK
# ─────────────────────────────────────────────

def evaluate_route(order_ids_seq, day):
    if not order_ids_seq: return None
    day_df = day_orders[day].set_index('ORDERID')
    total_cube = sum(day_df.loc[oid, 'CUBE'] for oid in order_ids_seq)
    if total_cube > VAN_CAPACITY: return None
    loc_seq = [DEPOT_ID] + [int(day_df.loc[oid, 'ZIPID']) for oid in order_ids_seq] + [DEPOT_ID]
    first_dist = get_dist(DEPOT_ID, loc_seq[1])
    dispatch_time = WINDOW_OPEN - first_dist / SPEED_MPM
    current_time = dispatch_time; drive_accum = 0.0; duty_accum = 0.0
    events = [{'loc': DEPOT_ID, 'arrive': None, 'depart': dispatch_time, 'type': 'depot'}]
    prev_loc = DEPOT_ID
    for i, loc in enumerate(loc_seq[1:], 1):
        oid = order_ids_seq[i-1] if i <= len(order_ids_seq) else None
        tt = get_dist(prev_loc, loc) / SPEED_MPM
        arrive_time = current_time + tt; drive_accum += tt; duty_accum += tt
        if oid is not None:
            if arrive_time < WINDOW_OPEN:
                duty_accum += (WINDOW_OPEN - arrive_time); arrive_time = WINDOW_OPEN
            if arrive_time > WINDOW_CLOSE: return None
            cube = day_df.loc[oid, 'CUBE']
            ul = unload_time(cube); depart_time = arrive_time + ul; duty_accum += ul
            if depart_time > WINDOW_CLOSE: return None
            events.append({'loc': loc, 'arrive': arrive_time, 'depart': depart_time,
                           'oid': oid, 'cube': cube, 'ul_time': ul,
                           'drive': drive_accum, 'duty': duty_accum})
        else:
            events.append({'loc': loc, 'arrive': arrive_time, 'depart': None, 'type': 'return'})
        if drive_accum > MAX_DRIVE_MIN or duty_accum > MAX_DUTY_MIN: return None
        current_time = depart_time if oid is not None else arrive_time
        prev_loc = loc
    total_dist = sum(get_dist(loc_seq[j], loc_seq[j+1]) for j in range(len(loc_seq)-1))
    return {'feasible': True, 'total_dist': total_dist, 'total_cube': total_cube,
            'drive_time': drive_accum, 'duty_time': duty_accum,
            'dispatch_time': dispatch_time, 'finish_time': current_time,
            'events': events, 'overnight': False}


# ─────────────────────────────────────────────
# OVERNIGHT ROUTE EVALUATION
# ─────────────────────────────────────────────

def evaluate_route_overnight(order_ids_day1, order_ids_day2, day1, day2):
    if not order_ids_day1 or not order_ids_day2: return None
    day1_df = day_orders[day1].set_index('ORDERID')
    day2_df = day_orders[day2].set_index('ORDERID')
    total_cube = (sum(day1_df.loc[oid, 'CUBE'] for oid in order_ids_day1) +
                  sum(day2_df.loc[oid, 'CUBE'] for oid in order_ids_day2))
    if total_cube > VAN_CAPACITY: return None

    loc_seq1 = [DEPOT_ID] + [int(day1_df.loc[oid, 'ZIPID']) for oid in order_ids_day1]
    first_dist = get_dist(DEPOT_ID, loc_seq1[1])
    dispatch_time = WINDOW_OPEN - first_dist / SPEED_MPM
    current_time = dispatch_time; drive1 = 0.0; duty1 = 0.0; prev_loc = DEPOT_ID

    for oid in order_ids_day1:
        loc = int(day1_df.loc[oid, 'ZIPID'])
        tt = get_dist(prev_loc, loc) / SPEED_MPM
        arrive = current_time + tt; drive1 += tt; duty1 += tt
        if arrive < WINDOW_OPEN: duty1 += (WINDOW_OPEN - arrive); arrive = WINDOW_OPEN
        if arrive > WINDOW_CLOSE: return None
        ul = unload_time(day1_df.loc[oid, 'CUBE'])
        depart = arrive + ul
        if depart > WINDOW_CLOSE: return None
        current_time = depart; duty1 += ul
        if drive1 > MAX_DRIVE_MIN or duty1 > MAX_DUTY_MIN: return None
        prev_loc = loc

    first_day2_loc = int(day2_df.loc[order_ids_day2[0], 'ZIPID'])
    time_to_day2 = get_dist(prev_loc, first_day2_loc) / SPEED_MPM
    can_continue = min(MAX_DRIVE_MIN - drive1, MAX_DUTY_MIN - duty1)
    if can_continue >= time_to_day2:
        drive1 += time_to_day2; duty1 += time_to_day2
        break_start = current_time + time_to_day2
    else:
        drive1 += can_continue; duty1 += can_continue
        break_start = current_time + can_continue
    break_end = break_start + BREAK_MIN
    day2_open = WINDOW_OPEN + 1440; day2_close = WINDOW_CLOSE + 1440
    arrive_d2 = break_end if can_continue >= time_to_day2 else break_end + (time_to_day2 - can_continue)
    if arrive_d2 > day2_close: return None
    current_time2 = max(arrive_d2, day2_open)
    drive2 = 0.0 if can_continue >= time_to_day2 else (time_to_day2 - can_continue)
    duty2 = current_time2 - break_end; prev_loc2 = first_day2_loc

    ul0 = unload_time(day2_df.loc[order_ids_day2[0], 'CUBE'])
    if current_time2 > day2_close: return None
    dep2 = current_time2 + ul0
    if dep2 > day2_close and len(order_ids_day2) > 1: return None
    current_time2 = dep2; duty2 += ul0

    for oid in order_ids_day2[1:]:
        loc = int(day2_df.loc[oid, 'ZIPID'])
        tt = get_dist(prev_loc2, loc) / SPEED_MPM
        arrive = current_time2 + tt; drive2 += tt; duty2 += tt
        if arrive < day2_open: duty2 += (day2_open - arrive); arrive = day2_open
        if arrive > day2_close: return None
        ul = unload_time(day2_df.loc[oid, 'CUBE'])
        dep = arrive + ul
        if dep > day2_close: return None
        current_time2 = dep; duty2 += ul
        if drive2 > MAX_DRIVE_MIN or duty2 > MAX_DUTY_MIN: return None
        prev_loc2 = loc

    ret_t = get_dist(prev_loc2, DEPOT_ID) / SPEED_MPM
    drive2 += ret_t; duty2 += ret_t
    if drive2 > MAX_DRIVE_MIN or duty2 > MAX_DUTY_MIN: return None
    finish_time = current_time2 + ret_t

    all_locs = ([DEPOT_ID] + [int(day1_df.loc[o, 'ZIPID']) for o in order_ids_day1] +
                [int(day2_df.loc[o, 'ZIPID']) for o in order_ids_day2] + [DEPOT_ID])
    total_dist = sum(get_dist(all_locs[j], all_locs[j+1]) for j in range(len(all_locs)-1))
    return {'feasible': True, 'overnight': True, 'day1': day1, 'day2': day2,
            'orders_day1': list(order_ids_day1), 'orders_day2': list(order_ids_day2),
            'total_dist': total_dist, 'total_cube': total_cube,
            'dispatch_time': dispatch_time, 'finish_time': finish_time,
            'drive1': drive1, 'duty1': duty1, 'drive2': drive2, 'duty2': duty2,
            'break_start': break_start, 'break_end': break_end}


# ═════════════════════════════════════════════
# CONSTRUCTION: NEAREST NEIGHBOR
# ═════════════════════════════════════════════

def nearest_neighbor_routes(day):
    day_df = day_orders[day].copy()
    unassigned = set(day_df['ORDERID'].tolist()); routes = []
    while unassigned:
        route = []; cap_used = 0; current_loc = DEPOT_ID
        drive_accum = 0.0; duty_accum = 0.0; current_time = None
        while unassigned:
            best_oid = None; best_dist = float('inf'); best_info = None
            for oid in unassigned:
                row = day_df[day_df['ORDERID'] == oid].iloc[0]
                cube = row['CUBE']
                if cap_used + cube > VAN_CAPACITY: continue
                loc = int(row['ZIPID']); d = get_dist(current_loc, loc); travel_t = d / SPEED_MPM
                if current_time is None:
                    arrive = WINDOW_OPEN; _drive = travel_t; _duty = travel_t
                else:
                    arrive = current_time + travel_t; _drive = drive_accum + travel_t; _duty = duty_accum + travel_t
                    if arrive < WINDOW_OPEN: _duty += (WINDOW_OPEN - arrive); arrive = WINDOW_OPEN
                if arrive > WINDOW_CLOSE: continue
                if _drive > MAX_DRIVE_MIN or _duty > MAX_DUTY_MIN: continue
                ul = unload_time(cube); depart = arrive + ul
                if depart > WINDOW_CLOSE: continue
                _duty_after = _duty + ul
                if _duty_after > MAX_DUTY_MIN: continue
                ret_t = get_dist(loc, DEPOT_ID) / SPEED_MPM
                if _drive + ret_t > MAX_DRIVE_MIN or _duty_after + ret_t > MAX_DUTY_MIN: continue
                if d < best_dist:
                    best_dist = d; best_oid = oid
                    best_info = (loc, arrive, depart, _drive, _duty_after, ul)
            if best_oid is None: break
            loc, arrive, depart, drive_accum, duty_accum, ul = best_info
            if current_time is None:
                dd = get_dist(DEPOT_ID, loc) / SPEED_MPM; drive_accum = dd; duty_accum = dd + ul
            current_time = depart; route.append(best_oid)
            cap_used += day_df[day_df['ORDERID'] == best_oid]['CUBE'].values[0]
            current_loc = loc; unassigned.remove(best_oid)
        if route: routes.append(route)
    return routes


# ═════════════════════════════════════════════
# CONSTRUCTION: CLARK-WRIGHT
# ═════════════════════════════════════════════

def clark_wright_routes(day):
    day_df = day_orders[day].copy().set_index('ORDERID')
    all_oids = day_df.index.tolist()
    if not all_oids: return []
    oid_zipid = {oid: int(day_df.loc[oid, 'ZIPID']) for oid in all_oids}
    oid_cube = {oid: day_df.loc[oid, 'CUBE'] for oid in all_oids}
    routes = {i: [oid] for i, oid in enumerate(all_oids)}
    route_of = {oid: i for i, oid in enumerate(all_oids)}
    nrid = len(all_oids); route_cube = {i: oid_cube[oid] for i, oid in enumerate(all_oids)}
    savings_list = []
    for a in range(len(all_oids)):
        oa = all_oids[a]; la = oid_zipid[oa]; da = get_dist(DEPOT_ID, la)
        for b in range(a+1, len(all_oids)):
            ob = all_oids[b]; lb = oid_zipid[ob]
            s = da + get_dist(DEPOT_ID, lb) - get_dist(la, lb)
            if s > 0: savings_list.append((s, oa, ob))
    savings_list.sort(key=lambda x: x[0], reverse=True)
    for s, oa, ob in savings_list:
        ra = route_of[oa]; rb = route_of[ob]
        if ra == rb: continue
        rta = routes[ra]; rtb = routes[rb]
        if route_cube[ra] + route_cube[rb] > VAN_CAPACITY: continue
        merged = None
        if rta[-1] == oa and rtb[0] == ob: merged = rta + rtb
        elif rtb[-1] == ob and rta[0] == oa: merged = rtb + rta
        elif rta[-1] == oa and rtb[-1] == ob: merged = rta + rtb[::-1]
        elif rta[0] == oa and rtb[0] == ob: merged = rta[::-1] + rtb
        else: continue
        result = evaluate_route(merged, day)
        if result is None: continue
        nid = nrid; nrid += 1
        routes[nid] = merged; route_cube[nid] = route_cube[ra] + route_cube[rb]
        for o in merged: route_of[o] = nid
        del routes[ra]; del routes[rb]; del route_cube[ra]; del route_cube[rb]
    return list(routes.values())


# ─────────────────────────────────────────────
# 2-OPT
# ─────────────────────────────────────────────

def route_total_dist(route, day):
    day_df = day_orders[day].set_index('ORDERID')
    ll = [DEPOT_ID] + [int(day_df.loc[o, 'ZIPID']) for o in route] + [DEPOT_ID]
    return sum(get_dist(ll[i], ll[i+1]) for i in range(len(ll)-1))

def two_opt_improve(route, day):
    if len(route) < 3: return route
    improved = True; best = route[:]; best_dist = route_total_dist(best, day)
    while improved:
        improved = False
        for i in range(1, len(best)-1):
            for j in range(i+1, len(best)):
                nr = best[:i] + best[i:j+1][::-1] + best[j+1:]
                nd = route_total_dist(nr, day)
                if nd < best_dist:
                    res = evaluate_route(nr, day)
                    if res: best = nr; best_dist = nd; improved = True
    return best


# ─────────────────────────────────────────────
# OVERNIGHT MERGING
# ─────────────────────────────────────────────

def attempt_overnight_merges(day_routes_dict):
    cpairs = [('Mon','Tue'),('Tue','Wed'),('Wed','Thu'),('Thu','Fri')]
    overnight_routes = []; consumed = set(); all_cands = []
    for d1, d2 in cpairs:
        for i, r1 in enumerate(day_routes_dict[d1]):
            if r1.get('overnight'): continue
            for j, r2 in enumerate(day_routes_dict[d2]):
                if r2.get('overnight'): continue
                res = evaluate_route_overnight(r1['orders'], r2['orders'], d1, d2)
                if res:
                    sav = r1['total_dist'] + r2['total_dist'] - res['total_dist']
                    if sav > 0:
                        all_cands.append({'i':i,'j':j,'day1':d1,'day2':d2,'savings':sav,'result':res})
    all_cands.sort(key=lambda c: c['savings'], reverse=True)
    for c in all_cands:
        k1 = (c['day1'], c['i']); k2 = (c['day2'], c['j'])
        if k1 in consumed or k2 in consumed: continue
        consumed.add(k1); consumed.add(k2)
        m = c['result']; m['orders'] = m['orders_day1'] + m['orders_day2']; m['day'] = c['day1']
        overnight_routes.append(m)
    final = []
    for day in DAY_ORDER:
        for i, r in enumerate(day_routes_dict[day]):
            if (day, i) not in consumed: final.append(r)
    final.extend(overnight_routes)
    return final


# ═════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════

def build_and_improve(raw_routes, day):
    results = []
    for route in raw_routes:
        imp = two_opt_improve(route, day)
        res = evaluate_route(imp, day)
        if res: res['day'] = day; res['orders'] = imp; results.append(res)
        else:
            res = evaluate_route(route, day)
            if res: res['day'] = day; res['orders'] = route; results.append(res)
    return results

def total_dist_of(rl): return sum(r['total_dist'] for r in rl)
def orders_covered(rl):
    s = set()
    for r in rl: s.update(r['orders'])
    return s


# ═════════════════════════════════════════════════
# PHASE 1: ROUTE CONSTRUCTION
# ═════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  PHASE 1: Construct & improve routes (new day assignments)")
print("=" * 60)

day_routes_dict = {}
for day in DAY_ORDER:
    day_oids = set(day_orders[day]['ORDERID'].tolist())
    print(f"\n  {day}  ({len(day_oids)} orders)")
    nn_raw = nearest_neighbor_routes(day)
    nn_r = build_and_improve(nn_raw, day); nn_d = total_dist_of(nn_r); nn_c = orders_covered(nn_r)
    print(f"    NN : {len(nn_r):>3} routes, {nn_d:>8,.1f} miles")
    cw_raw = clark_wright_routes(day)
    cw_r = build_and_improve(cw_raw, day); cw_d = total_dist_of(cw_r); cw_c = orders_covered(cw_r)
    print(f"    CW : {len(cw_r):>3} routes, {cw_d:>8,.1f} miles")
    nn_ok = nn_c == day_oids; cw_ok = cw_c == day_oids
    if cw_ok and (not nn_ok or cw_d < nn_d):
        day_routes_dict[day] = cw_r; w = "CW"
    else:
        day_routes_dict[day] = nn_r; w = "NN"
    print(f"    >>> Selected: {w}")

all_before = []; [all_before.extend(day_routes_dict[d]) for d in DAY_ORDER]
miles_before = total_dist_of(all_before)
print(f"\n{'=' * 60}")
print(f"  BEFORE OVERNIGHT : {len(all_before)} routes, {miles_before:,.1f} miles")
print(f"{'=' * 60}")

# ═════════════════════════════════════════════════
# PHASE 2: OVERNIGHT MERGING
# ═════════════════════════════════════════════════
print("\n  PHASE 2: Overnight route merges...")
all_routes = attempt_overnight_merges(day_routes_dict)
overnight_count = sum(1 for r in all_routes if r.get('overnight'))
total_miles = total_dist_of(all_routes)
savings_on = miles_before - total_miles
print(f"  Overnight routes: {overnight_count}, saved {savings_on:,.1f} miles")

covered = set(); [covered.update(r['orders']) for r in all_routes]
all_oids = set(orders['ORDERID'].tolist())
print(f"  Orders covered: {len(covered)} / {len(all_oids)}")
if all_oids - covered: print(f"  *** MISSING: {all_oids - covered}")


# ─────────────────────────────────────────────
# DRIVER & VEHICLE ANALYSIS
# ─────────────────────────────────────────────
print("\n─── DRIVER & VEHICLE REQUIREMENT ANALYSIS ───")
day_offsets = {'Mon': 0, 'Tue': 1440, 'Wed': 2880, 'Thu': 4320, 'Fri': 5760}
def abs_time(d, m): return day_offsets[d] + m

route_windows = []
for i, r in enumerate(all_routes):
    d = r['day']; da = abs_time(d, r['dispatch_time'])
    fa = (day_offsets[d] + r['finish_time']) if r.get('overnight') else abs_time(d, r['finish_time'])
    route_windows.append({'route_idx': i, 'day': d, 'dispatch_abs': da, 'finish_abs': fa})

evts = []; [(evts.append((rw['dispatch_abs'],+1)), evts.append((rw['finish_abs'],-1))) for rw in route_windows]
evts.sort(); conc = mx = 0
for _, delta in evts: conc += delta; mx = max(mx, conc)
min_vehicles = mx; print(f"  Minimum vehicles: {min_vehicles}")

rws = sorted(route_windows, key=lambda x: x['dispatch_abs']); drvrs = []
for rw in rws:
    aa = rw['finish_abs'] + BREAK_MIN; assigned = False
    for idx, av in enumerate(drvrs):
        if av <= rw['dispatch_abs']: drvrs[idx] = aa; assigned = True; break
    if not assigned: drvrs.append(aa)
min_drivers = len(drvrs); print(f"  Minimum drivers : {min_drivers}")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n─── ROUTE SUMMARY BY DAY ───")
print(f"{'Day':<6} {'Routes':>7} {'Ovrnt':>6} {'Orders':>7} {'Miles':>10}")
print("-" * 42)
for day in DAY_ORDER:
    dr = [r for r in all_routes if r['day'] == day]
    print(f"{day:<6} {len(dr):>7} {sum(1 for r in dr if r.get('overnight')):>6} "
          f"{sum(len(r['orders']) for r in dr):>7} {sum(r['total_dist'] for r in dr):>10,.1f}")
print("-" * 42)
ton = sum(1 for r in all_routes if r.get('overnight'))
print(f"{'TOTAL':<6} {len(all_routes):>7} {ton:>6} {len(covered):>7} {total_miles:>10,.1f}")

print(f"\n{'=' * 60}")
print(f"  TOTAL WEEKLY MILES       : {total_miles:>10,.1f}")
print(f"  TOTAL ANNUAL MILES (x52) : {total_miles*52:>10,.1f}")
print(f"  OVERNIGHT ROUTES         : {ton:>10}")
print(f"  MINIMUM VEHICLES         : {min_vehicles:>10}")
print(f"  MINIMUM DRIVERS          : {min_drivers:>10}")
print(f"{'=' * 60}")


