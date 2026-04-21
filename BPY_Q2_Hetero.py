"""
NHG Vehicle Routing Problem Solver
Question 2: Heterogeneous fleet (Van + Straight Truck), CVRP with DOT + overnight routes
Heuristics:
  Construction  — Nearest-Neighbor  AND  Clark-Wright Savings (best of both kept)
  Improvement   — 2-opt intra-route
  Multi-day     — Overnight route merging (same vehicle type only)
"""

import pandas as pd
import numpy as np
from itertools import combinations
from copy import deepcopy
from collections import defaultdict

# ─────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────
# Van
VAN_CAPACITY    = 3200
VAN_UNLOAD_RATE = 0.03

# Straight Truck
ST_CAPACITY     = 1400
ST_UNLOAD_RATE  = 0.043

MIN_UNLOAD      = 30
SPEED           = 40
SPEED_MPM       = 40 / 60

MAX_DRIVE_HR    = 11
MAX_DUTY_HR     = 14
BREAK_HR        = 10
MAX_DRIVE_MIN   = MAX_DRIVE_HR  * 60
MAX_DUTY_MIN    = MAX_DUTY_HR   * 60
BREAK_MIN       = BREAK_HR      * 60

WINDOW_OPEN     = 8  * 60
WINDOW_CLOSE    = 18 * 60

DAY_ORDER       = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("Loading data...")
orders_raw = pd.read_excel('deliveries.xlsx', sheet_name='OrderTable')
locs       = pd.read_excel('deliveries.xlsx', sheet_name='LocationTable')
dist_raw   = pd.read_excel('distances.xlsx', header=1, index_col=1)

orders = orders_raw[orders_raw['ORDERID'] != 0].copy()
orders['CUBE'] = pd.to_numeric(orders['CUBE'], errors='coerce')
orders['ST_REQUIRED'] = orders['ST required?'].str.strip().str.lower() == 'yes'

locs = locs.dropna(subset=['ZIP', 'ZIPID'])
locs['ZIP']   = locs['ZIP'].astype(int)
locs['ZIPID'] = locs['ZIPID'].astype(int)
zip_to_id = dict(zip(locs['ZIP'], locs['ZIPID']))

dist_matrix = {}
for row_id, row in dist_raw.iterrows():
    dist_matrix[int(row_id)] = {}
    for col_id in dist_raw.columns[1:]:
        dist_matrix[int(row_id)][int(col_id)] = int(row[col_id])

DEPOT_ID = 20

def get_dist(id1, id2):
    if id1 == id2:
        return 0
    return dist_matrix[id1][id2]

def unload_time(cube, vehicle_type):
    rate = ST_UNLOAD_RATE if vehicle_type == 'ST' else VAN_UNLOAD_RATE
    return max(MIN_UNLOAD, rate * cube)

def get_capacity(vehicle_type):
    return ST_CAPACITY if vehicle_type == 'ST' else VAN_CAPACITY

# Build per-day order lists — separate ST-required and Van-eligible
day_orders_all = {}
day_orders_st  = {}
day_orders_van = {}

for day in DAY_ORDER:
    day_df = orders[orders['DayOfWeek'] == day].copy()
    day_df['ZIPID'] = day_df['TOZIP'].map(lambda z: zip_to_id.get(int(z), None))
    day_orders_all[day] = day_df.reset_index(drop=True)
    day_orders_st[day]  = day_df[day_df['ST_REQUIRED'] == True].reset_index(drop=True)
    day_orders_van[day] = day_df[day_df['ST_REQUIRED'] == False].reset_index(drop=True)

print(f"  ST-required orders: {orders['ST_REQUIRED'].sum()}")
print(f"  Van-eligible orders: {(~orders['ST_REQUIRED']).sum()}")


# ─────────────────────────────────────────────
# ROUTE FEASIBILITY CHECK (single-day, vehicle-aware)
# ─────────────────────────────────────────────

def evaluate_route(order_ids_seq, day, vehicle_type='Van'):
    if not order_ids_seq:
        return None
    day_df = day_orders_all[day].set_index('ORDERID')
    capacity = get_capacity(vehicle_type)
    total_cube = sum(day_df.loc[oid, 'CUBE'] for oid in order_ids_seq)
    if total_cube > capacity:
        return None
    loc_seq = [DEPOT_ID] + [int(day_df.loc[oid, 'ZIPID']) for oid in order_ids_seq] + [DEPOT_ID]
    first_dist = get_dist(DEPOT_ID, loc_seq[1])
    dispatch_time = WINDOW_OPEN - first_dist / SPEED_MPM
    current_time = dispatch_time
    drive_accum = 0.0
    duty_accum = 0.0
    events = [{'loc': DEPOT_ID, 'arrive': None, 'depart': dispatch_time, 'type': 'depot'}]
    prev_loc = DEPOT_ID
    for i, loc in enumerate(loc_seq[1:], 1):
        oid = order_ids_seq[i-1] if i <= len(order_ids_seq) else None
        travel_dist = get_dist(prev_loc, loc)
        travel_time = travel_dist / SPEED_MPM
        arrive_time = current_time + travel_time
        drive_accum += travel_time
        duty_accum += travel_time
        if oid is not None:
            if arrive_time < WINDOW_OPEN:
                duty_accum += (WINDOW_OPEN - arrive_time)
                arrive_time = WINDOW_OPEN
            if arrive_time > WINDOW_CLOSE:
                return None
            cube = day_df.loc[oid, 'CUBE']
            ul_time = unload_time(cube, vehicle_type)
            depart_time = arrive_time + ul_time
            duty_accum += ul_time
            if depart_time > WINDOW_CLOSE:
                return None
            events.append({'loc': loc, 'arrive': arrive_time, 'depart': depart_time,
                           'oid': oid, 'cube': cube, 'ul_time': ul_time,
                           'drive': drive_accum, 'duty': duty_accum})
        else:
            events.append({'loc': loc, 'arrive': arrive_time, 'depart': None, 'type': 'return'})
        if drive_accum > MAX_DRIVE_MIN or duty_accum > MAX_DUTY_MIN:
            return None
        current_time = depart_time if oid is not None else arrive_time
        prev_loc = loc
    total_dist = sum(get_dist(loc_seq[j], loc_seq[j+1]) for j in range(len(loc_seq)-1))
    return {
        'feasible': True, 'total_dist': total_dist, 'total_cube': total_cube,
        'drive_time': drive_accum, 'duty_time': duty_accum,
        'dispatch_time': dispatch_time, 'finish_time': current_time,
        'events': events, 'overnight': False, 'vehicle_type': vehicle_type
    }


# ─────────────────────────────────────────────
# OVERNIGHT ROUTE EVALUATION (vehicle-aware)
# ─────────────────────────────────────────────

def evaluate_route_overnight(order_ids_day1, order_ids_day2, day1, day2, vehicle_type='Van'):
    if not order_ids_day1 or not order_ids_day2:
        return None
    day1_df = day_orders_all[day1].set_index('ORDERID')
    day2_df = day_orders_all[day2].set_index('ORDERID')
    capacity = get_capacity(vehicle_type)

    total_cube = (sum(day1_df.loc[oid, 'CUBE'] for oid in order_ids_day1) +
                  sum(day2_df.loc[oid, 'CUBE'] for oid in order_ids_day2))
    if total_cube > capacity:
        return None

    loc_seq1 = [DEPOT_ID] + [int(day1_df.loc[oid, 'ZIPID']) for oid in order_ids_day1]
    first_dist = get_dist(DEPOT_ID, loc_seq1[1])
    dispatch_time = WINDOW_OPEN - first_dist / SPEED_MPM
    current_time = dispatch_time
    drive1 = 0.0; duty1 = 0.0
    prev_loc = DEPOT_ID

    for oid in order_ids_day1:
        loc = int(day1_df.loc[oid, 'ZIPID'])
        tt = get_dist(prev_loc, loc) / SPEED_MPM
        arrive = current_time + tt
        drive1 += tt; duty1 += tt
        if arrive < WINDOW_OPEN:
            duty1 += (WINDOW_OPEN - arrive); arrive = WINDOW_OPEN
        if arrive > WINDOW_CLOSE:
            return None
        ul = unload_time(day1_df.loc[oid, 'CUBE'], vehicle_type)
        depart = arrive + ul
        if depart > WINDOW_CLOSE:
            return None
        current_time = depart; duty1 += ul
        if drive1 > MAX_DRIVE_MIN or duty1 > MAX_DUTY_MIN:
            return None
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
    day2_open = WINDOW_OPEN + 1440
    day2_close = WINDOW_CLOSE + 1440

    if can_continue >= time_to_day2:
        arrive_d2_first = break_end
    else:
        arrive_d2_first = break_end + (time_to_day2 - can_continue)

    if arrive_d2_first > day2_close:
        return None

    current_time2 = max(arrive_d2_first, day2_open)
    drive2 = 0.0 if can_continue >= time_to_day2 else (time_to_day2 - can_continue)
    duty2 = current_time2 - break_end
    prev_loc2 = first_day2_loc

    ul0 = unload_time(day2_df.loc[order_ids_day2[0], 'CUBE'], vehicle_type)
    if current_time2 > day2_close:
        return None
    depart2 = current_time2 + ul0
    if depart2 > day2_close and len(order_ids_day2) > 1:
        return None
    current_time2 = depart2; duty2 += ul0

    for oid in order_ids_day2[1:]:
        loc = int(day2_df.loc[oid, 'ZIPID'])
        tt = get_dist(prev_loc2, loc) / SPEED_MPM
        arrive = current_time2 + tt
        drive2 += tt; duty2 += tt
        if arrive < day2_open:
            duty2 += (day2_open - arrive); arrive = day2_open
        if arrive > day2_close:
            return None
        ul = unload_time(day2_df.loc[oid, 'CUBE'], vehicle_type)
        depart = arrive + ul
        if depart > day2_close:
            return None
        current_time2 = depart; duty2 += ul
        if drive2 > MAX_DRIVE_MIN or duty2 > MAX_DUTY_MIN:
            return None
        prev_loc2 = loc

    ret_t = get_dist(prev_loc2, DEPOT_ID) / SPEED_MPM
    drive2 += ret_t; duty2 += ret_t
    if drive2 > MAX_DRIVE_MIN or duty2 > MAX_DUTY_MIN:
        return None
    finish_time = current_time2 + ret_t

    all_locs = ([DEPOT_ID] +
                [int(day1_df.loc[oid, 'ZIPID']) for oid in order_ids_day1] +
                [int(day2_df.loc[oid, 'ZIPID']) for oid in order_ids_day2] +
                [DEPOT_ID])
    total_dist = sum(get_dist(all_locs[j], all_locs[j+1]) for j in range(len(all_locs)-1))

    return {
        'feasible': True, 'overnight': True,
        'day1': day1, 'day2': day2,
        'orders_day1': list(order_ids_day1),
        'orders_day2': list(order_ids_day2),
        'total_dist': total_dist, 'total_cube': total_cube,
        'dispatch_time': dispatch_time, 'finish_time': finish_time,
        'drive1': drive1, 'duty1': duty1,
        'drive2': drive2, 'duty2': duty2,
        'break_start': break_start, 'break_end': break_end,
        'vehicle_type': vehicle_type
    }


# ═════════════════════════════════════════════
# CONSTRUCTION: NEAREST NEIGHBOR (vehicle-aware)
# ═════════════════════════════════════════════

def nearest_neighbor_routes(day, order_subset_df, vehicle_type='Van'):
    day_df = order_subset_df.copy()
    capacity = get_capacity(vehicle_type)
    unassigned = set(day_df['ORDERID'].tolist())
    routes = []
    while unassigned:
        route = []
        cap_used = 0
        current_loc = DEPOT_ID
        drive_accum = 0.0; duty_accum = 0.0
        current_time = None
        while unassigned:
            best_oid = None; best_dist = float('inf'); best_info = None
            for oid in unassigned:
                row = day_df[day_df['ORDERID'] == oid].iloc[0]
                cube = row['CUBE']
                if cap_used + cube > capacity:
                    continue
                loc = int(row['ZIPID'])
                d = get_dist(current_loc, loc)
                travel_t = d / SPEED_MPM
                if current_time is None:
                    arrive = WINDOW_OPEN; _drive = travel_t; _duty = travel_t
                else:
                    arrive = current_time + travel_t
                    _drive = drive_accum + travel_t; _duty = duty_accum + travel_t
                    if arrive < WINDOW_OPEN:
                        _duty += (WINDOW_OPEN - arrive); arrive = WINDOW_OPEN
                if arrive > WINDOW_CLOSE:
                    continue
                if _drive > MAX_DRIVE_MIN or _duty > MAX_DUTY_MIN:
                    continue
                ul = unload_time(cube, vehicle_type)
                depart = arrive + ul
                if depart > WINDOW_CLOSE:
                    continue
                _duty_after = _duty + ul
                if _duty_after > MAX_DUTY_MIN:
                    continue
                ret_t = get_dist(loc, DEPOT_ID) / SPEED_MPM
                if _drive + ret_t > MAX_DRIVE_MIN or _duty_after + ret_t > MAX_DUTY_MIN:
                    continue
                if d < best_dist:
                    best_dist = d; best_oid = oid
                    best_info = (loc, arrive, depart, _drive, _duty_after, ul)
            if best_oid is None:
                break
            loc, arrive, depart, drive_accum, duty_accum, ul = best_info
            if current_time is None:
                dd = get_dist(DEPOT_ID, loc) / SPEED_MPM
                drive_accum = dd; duty_accum = dd + ul
            current_time = depart
            route.append(best_oid)
            cap_used += day_df[day_df['ORDERID'] == best_oid]['CUBE'].values[0]
            current_loc = loc
            unassigned.remove(best_oid)
        if route:
            routes.append(route)
    return routes


# ═════════════════════════════════════════════
# CONSTRUCTION: CLARK-WRIGHT (vehicle-aware)
# ═════════════════════════════════════════════

def clark_wright_routes(day, order_subset_df, vehicle_type='Van'):
    day_df = order_subset_df.copy().set_index('ORDERID')
    all_oids = day_df.index.tolist()
    if not all_oids:
        return []
    capacity = get_capacity(vehicle_type)
    oid_zipid = {oid: int(day_df.loc[oid, 'ZIPID']) for oid in all_oids}
    oid_cube = {oid: day_df.loc[oid, 'CUBE'] for oid in all_oids}
    routes = {i: [oid] for i, oid in enumerate(all_oids)}
    route_of = {oid: i for i, oid in enumerate(all_oids)}
    next_route_id = len(all_oids)
    route_cube = {i: oid_cube[oid] for i, oid in enumerate(all_oids)}
    savings_list = []
    for idx_a in range(len(all_oids)):
        oid_a = all_oids[idx_a]
        loc_a = oid_zipid[oid_a]
        d_depot_a = get_dist(DEPOT_ID, loc_a)
        for idx_b in range(idx_a + 1, len(all_oids)):
            oid_b = all_oids[idx_b]
            loc_b = oid_zipid[oid_b]
            s = d_depot_a + get_dist(DEPOT_ID, loc_b) - get_dist(loc_a, loc_b)
            if s > 0:
                savings_list.append((s, oid_a, oid_b))
    savings_list.sort(key=lambda x: x[0], reverse=True)
    for s, oid_a, oid_b in savings_list:
        ra = route_of[oid_a]; rb = route_of[oid_b]
        if ra == rb:
            continue
        route_a = routes[ra]; route_b = routes[rb]
        if route_cube[ra] + route_cube[rb] > capacity:
            continue
        merged = None
        if route_a[-1] == oid_a and route_b[0] == oid_b:
            merged = route_a + route_b
        elif route_b[-1] == oid_b and route_a[0] == oid_a:
            merged = route_b + route_a
        elif route_a[-1] == oid_a and route_b[-1] == oid_b:
            merged = route_a + route_b[::-1]
        elif route_a[0] == oid_a and route_b[0] == oid_b:
            merged = route_a[::-1] + route_b
        else:
            continue
        result = evaluate_route(merged, day, vehicle_type)
        if result is None:
            continue
        new_id = next_route_id; next_route_id += 1
        routes[new_id] = merged
        route_cube[new_id] = route_cube[ra] + route_cube[rb]
        for oid in merged:
            route_of[oid] = new_id
        del routes[ra]; del routes[rb]
        del route_cube[ra]; del route_cube[rb]
    return list(routes.values())


# ─────────────────────────────────────────────
# 2-OPT (vehicle-aware)
# ─────────────────────────────────────────────

def route_total_dist(route, day):
    day_df = day_orders_all[day].set_index('ORDERID')
    locs_list = [DEPOT_ID] + [int(day_df.loc[oid, 'ZIPID']) for oid in route] + [DEPOT_ID]
    return sum(get_dist(locs_list[i], locs_list[i+1]) for i in range(len(locs_list)-1))

def two_opt_improve(route, day, vehicle_type='Van'):
    if len(route) < 3:
        return route
    improved = True
    best = route[:]; best_dist = route_total_dist(best, day)
    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                new_dist = route_total_dist(new_route, day)
                if new_dist < best_dist:
                    result = evaluate_route(new_route, day, vehicle_type)
                    if result:
                        best = new_route; best_dist = new_dist; improved = True
    return best


# ─────────────────────────────────────────────
# OVERNIGHT MERGING (same vehicle type only)
# ─────────────────────────────────────────────

def attempt_overnight_merges(day_routes_dict):
    consecutive_pairs = [('Mon', 'Tue'), ('Tue', 'Wed'), ('Wed', 'Thu'), ('Thu', 'Fri')]
    overnight_routes = []
    consumed = set()
    all_candidates = []

    for day1, day2 in consecutive_pairs:
        routes_d1 = day_routes_dict[day1]
        routes_d2 = day_routes_dict[day2]
        for i, r1 in enumerate(routes_d1):
            if r1.get('overnight'):
                continue
            for j, r2 in enumerate(routes_d2):
                if r2.get('overnight'):
                    continue
                # Only merge same vehicle type
                if r1.get('vehicle_type', 'Van') != r2.get('vehicle_type', 'Van'):
                    continue
                vtype = r1.get('vehicle_type', 'Van')
                result = evaluate_route_overnight(
                    r1['orders'], r2['orders'], day1, day2, vtype)
                if result is not None:
                    savings = r1['total_dist'] + r2['total_dist'] - result['total_dist']
                    if savings > 0:
                        all_candidates.append({
                            'i': i, 'j': j, 'day1': day1, 'day2': day2,
                            'savings': savings, 'result': result
                        })

    all_candidates.sort(key=lambda c: c['savings'], reverse=True)
    for c in all_candidates:
        key1 = (c['day1'], c['i']); key2 = (c['day2'], c['j'])
        if key1 in consumed or key2 in consumed:
            continue
        consumed.add(key1); consumed.add(key2)
        merged = c['result']
        merged['orders'] = merged['orders_day1'] + merged['orders_day2']
        merged['day'] = c['day1']
        overnight_routes.append(merged)

    final_routes = []
    for day in DAY_ORDER:
        for i, r in enumerate(day_routes_dict[day]):
            if (day, i) not in consumed:
                final_routes.append(r)
    final_routes.extend(overnight_routes)
    return final_routes


# ═════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════

def build_and_improve(raw_routes, day, vehicle_type='Van'):
    results = []
    for route in raw_routes:
        improved = two_opt_improve(route, day, vehicle_type)
        res = evaluate_route(improved, day, vehicle_type)
        if res:
            res['day'] = day; res['orders'] = improved
            results.append(res)
        else:
            res = evaluate_route(route, day, vehicle_type)
            if res:
                res['day'] = day; res['orders'] = route
                results.append(res)
    return results

def total_dist_of(route_list):
    return sum(r['total_dist'] for r in route_list)

def orders_covered(route_list):
    s = set()
    for r in route_list:
        s.update(r['orders'])
    return s


# ═════════════════════════════════════════════════
# MAIN SOLVING LOOP
# ═════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  PHASE 1: Construct & improve single-day routes")
print("  (Straight Truck routes + Van routes separately)")
print("=" * 60)

day_routes_dict = {}

for day in DAY_ORDER:
    st_df  = day_orders_st[day]
    van_df = day_orders_van[day]
    print(f"\n  {day}  (ST: {len(st_df)} orders, Van-eligible: {len(van_df)} orders)")

    # ── Straight Truck routes ──
    st_results = []
    if len(st_df) > 0:
        nn_st = nearest_neighbor_routes(day, st_df, 'ST')
        nn_st_r = build_and_improve(nn_st, day, 'ST')
        nn_st_dist = total_dist_of(nn_st_r)

        cw_st = clark_wright_routes(day, st_df, 'ST')
        cw_st_r = build_and_improve(cw_st, day, 'ST')
        cw_st_dist = total_dist_of(cw_st_r)

        st_oids = set(st_df['ORDERID'].tolist())
        nn_ok = orders_covered(nn_st_r) == st_oids
        cw_ok = orders_covered(cw_st_r) == st_oids

        if cw_ok and (not nn_ok or cw_st_dist < nn_st_dist):
            st_results = cw_st_r; st_winner = "CW"
        else:
            st_results = nn_st_r; st_winner = "NN"
        print(f"    ST routes: {len(st_results):>3} routes, {total_dist_of(st_results):>8,.1f} miles ({st_winner})")

    # ── Van routes ──
    van_results = []
    if len(van_df) > 0:
        nn_van = nearest_neighbor_routes(day, van_df, 'Van')
        nn_van_r = build_and_improve(nn_van, day, 'Van')
        nn_van_dist = total_dist_of(nn_van_r)

        cw_van = clark_wright_routes(day, van_df, 'Van')
        cw_van_r = build_and_improve(cw_van, day, 'Van')
        cw_van_dist = total_dist_of(cw_van_r)

        van_oids = set(van_df['ORDERID'].tolist())
        nn_ok = orders_covered(nn_van_r) == van_oids
        cw_ok = orders_covered(cw_van_r) == van_oids

        if cw_ok and (not nn_ok or cw_van_dist < nn_van_dist):
            van_results = cw_van_r; van_winner = "CW"
        else:
            van_results = nn_van_r; van_winner = "NN"
        print(f"    Van routes: {len(van_results):>3} routes, {total_dist_of(van_results):>8,.1f} miles ({van_winner})")

    day_routes_dict[day] = st_results + van_results
    day_total = total_dist_of(day_routes_dict[day])
    print(f"    Day total: {len(day_routes_dict[day]):>3} routes, {day_total:>8,.1f} miles")


# ── Report BEFORE overnight ──
all_before = []
for day in DAY_ORDER:
    all_before.extend(day_routes_dict[day])
miles_before = total_dist_of(all_before)
print(f"\n{'=' * 60}")
print(f"  BEFORE OVERNIGHT MERGING")
print(f"  Total routes       : {len(all_before)}")
print(f"  Total weekly miles : {miles_before:>10,.1f}")
print(f"{'=' * 60}")


# ═════════════════════════════════════════════════
# PHASE 2: Overnight route merging
# ═════════════════════════════════════════════════
print("\n  PHASE 2: Attempting overnight route merges (same vehicle type)...")
all_routes = attempt_overnight_merges(day_routes_dict)

overnight_count = sum(1 for r in all_routes if r.get('overnight', False))
total_miles = total_dist_of(all_routes)
savings = miles_before - total_miles

print(f"  Overnight routes created : {overnight_count}")
print(f"  Miles saved              : {savings:,.1f}")

# ── Verify coverage ──
covered = set()
for r in all_routes:
    covered.update(r['orders'])
all_order_ids = set(orders['ORDERID'].tolist())
print(f"\n  Orders covered: {len(covered)} / {len(all_order_ids)}")
if all_order_ids - covered:
    print(f"  *** MISSING: {all_order_ids - covered}")

# Vehicle counts
n_van_routes = sum(1 for r in all_routes if r.get('vehicle_type', 'Van') == 'Van')
n_st_routes  = sum(1 for r in all_routes if r.get('vehicle_type', 'Van') == 'ST')
van_miles = sum(r['total_dist'] for r in all_routes if r.get('vehicle_type', 'Van') == 'Van')
st_miles  = sum(r['total_dist'] for r in all_routes if r.get('vehicle_type', 'Van') == 'ST')


# ─────────────────────────────────────────────
# DRIVER & VEHICLE ANALYSIS
# ─────────────────────────────────────────────
print("\n─── DRIVER & VEHICLE REQUIREMENT ANALYSIS ───")

day_offsets = {'Mon': 0, 'Tue': 1440, 'Wed': 2880, 'Thu': 4320, 'Fri': 5760}

def abs_time(day, minutes_from_midnight):
    return day_offsets[day] + minutes_from_midnight

route_windows = []
for i, r in enumerate(all_routes):
    d = r['day']
    dispatch_abs = abs_time(d, r['dispatch_time'])
    if r.get('overnight', False):
        finish_abs = day_offsets[d] + r['finish_time']
    else:
        finish_abs = abs_time(d, r['finish_time'])
    route_windows.append({
        'route_idx': i, 'day': d,
        'dispatch_abs': dispatch_abs, 'finish_abs': finish_abs,
        'dist': r['total_dist'], 'vehicle_type': r.get('vehicle_type', 'Van')
    })

events_list = []
for rw in route_windows:
    events_list.append((rw['dispatch_abs'], +1))
    events_list.append((rw['finish_abs'],   -1))
events_list.sort()
concurrent = max_concurrent = 0
for _, delta in events_list:
    concurrent += delta
    max_concurrent = max(max_concurrent, concurrent)
min_vehicles = max_concurrent
print(f"  Minimum vehicles required : {min_vehicles} (total)")

# Count by type
rw_van = [rw for rw in route_windows if rw['vehicle_type'] == 'Van']
rw_st  = [rw for rw in route_windows if rw['vehicle_type'] == 'ST']

for label, rws in [('Vans', rw_van), ('Straight Trucks', rw_st)]:
    evts = []
    for rw in rws:
        evts.append((rw['dispatch_abs'], +1))
        evts.append((rw['finish_abs'], -1))
    evts.sort()
    c = mx = 0
    for _, d in evts:
        c += d; mx = max(mx, c)
    print(f"    {label}: {mx}")

rw_sorted = sorted(route_windows, key=lambda x: x['dispatch_abs'])
drivers = []
for rw in rw_sorted:
    avail_after = rw['finish_abs'] + BREAK_MIN
    assigned = False
    for idx, avail in enumerate(drivers):
        if avail <= rw['dispatch_abs']:
            drivers[idx] = avail_after; assigned = True; break
    if not assigned:
        drivers.append(avail_after)
min_drivers = len(drivers)
print(f"  Minimum drivers required  : {min_drivers}")


# ─────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n─── ROUTE SUMMARY BY DAY ───")
print(f"{'Day':<6} {'Routes':>7} {'Van':>5} {'ST':>5} {'Ovrnt':>6} {'Orders':>7} {'Miles':>10}")
print("-" * 52)

for day in DAY_ORDER:
    day_r = [r for r in all_routes if r['day'] == day]
    n_routes = len(day_r)
    n_van = sum(1 for r in day_r if r.get('vehicle_type') == 'Van')
    n_st = sum(1 for r in day_r if r.get('vehicle_type') == 'ST')
    n_on = sum(1 for r in day_r if r.get('overnight'))
    n_orders = sum(len(r['orders']) for r in day_r)
    miles = sum(r['total_dist'] for r in day_r)
    print(f"{day:<6} {n_routes:>7} {n_van:>5} {n_st:>5} {n_on:>6} {n_orders:>7} {miles:>10,.1f}")

print("-" * 52)
total_on = sum(1 for r in all_routes if r.get('overnight'))
print(f"{'TOTAL':<6} {len(all_routes):>7} {n_van_routes:>5} {n_st_routes:>5} {total_on:>6} "
      f"{sum(len(r['orders']) for r in all_routes):>7} {total_miles:>10,.1f}")

print(f"\n{'=' * 60}")
print(f"  TOTAL WEEKLY MILES       : {total_miles:>10,.1f}")
print(f"  TOTAL ANNUAL MILES (x52) : {total_miles*52:>10,.1f}")
print(f"  VAN ROUTES / MILES       : {n_van_routes:>5} / {van_miles:>,.1f}")
print(f"  ST  ROUTES / MILES       : {n_st_routes:>5} / {st_miles:>,.1f}")
print(f"  OVERNIGHT ROUTES         : {total_on:>10}")
print(f"  MILES SAVED (overnight)  : {savings:>10,.1f}")
print(f"  MINIMUM VEHICLES         : {min_vehicles:>10}")
print(f"  MINIMUM DRIVERS          : {min_drivers:>10}")
print(f"{'=' * 60}")
