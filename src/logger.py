import csv
import os
from dataclasses import dataclass, asdict
from typing import List, Optional


NEIGHBORHOOD_NAMES = {
    1: "or_opt_1",
    2: "or_opt_2",
    3: "or_opt_3",
    4: "double_bridge",
    5: "double_double_bridge",
}


@dataclass
class LogEntry:
    instance_name: str
    k_max: int
    max_no_improve: int
    init_method:        str
    seed:               int
    optimal:            float

    # --- iteration level ---
    iteration:          int
    k:                  int
    neighborhood:       str
    cost_before:        float
    cost_after_shake:   float
    cost_after:         float
    best_cost:          float
    gap:                float
    shake_delta:        float
    ls_recovery:        float
    improved:           bool
    shaking_occurred:   bool
    time_elapsed:       float
    current_tour:       str
    best_tour:          str


class Logger:

    def __init__(
            self,
            instance_name:  str,
            run_id:         int   = 1,
            optimal:        float = None,
            k_max:          int   = 5,
            max_no_improve: int   = 100,
            init_method:    str   = "nearest_neighbor",
    ):
        self.instance_name  = instance_name
        self.run_id         = run_id
        self.optimal        = optimal
        self.k_max          = k_max
        self.max_no_improve = max_no_improve
        self.init_method    = init_method
        self.entries: List[LogEntry] = []

    def _compute_gap(self, cost: float) -> Optional[float]:
        if self.optimal is not None and self.optimal > 0:
            gap = round((cost - self.optimal) / self.optimal * 100, 4)
            return 0.0 if abs(gap) < 1e-6 else gap
        return None

    def log(
            self,
            iteration:          int,
            k:                  int,
            cost_before:        float,
            cost_after_shake:   float,
            cost_after:         float,
            best_cost:          float,
            improved:           bool,
            shaking_occurred:   bool,
            time_elapsed:       float,
            current_tour:       list,
            best_tour:          list,
    ):
        entry = LogEntry(
            instance_name=self.instance_name,
            k_max=self.k_max,
            max_no_improve=self.max_no_improve,
            init_method=self.init_method,
            seed=self.run_id,
            optimal=self.optimal,
            iteration=iteration,
            k=k,
            neighborhood=NEIGHBORHOOD_NAMES.get(k, "unknown"),
            cost_before=round(cost_before, 4),
            cost_after_shake=round(cost_after_shake, 4),
            cost_after=round(cost_after, 4),
            best_cost=round(best_cost, 4),
            gap=self._compute_gap(best_cost),
            shake_delta=round(cost_after_shake - cost_before, 4),
            ls_recovery=round(cost_after_shake - cost_after, 4),
            improved=improved,
            shaking_occurred=shaking_occurred,
            time_elapsed=round(time_elapsed, 4),
            current_tour=str(current_tour),
            best_tour=str(best_tour),
        )
        self.entries.append(entry)

    def save(self):
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        filename = (f"{self.instance_name}_kmax{self.k_max}_"
                    f"noimprove{self.max_no_improve}_"
                    f"{self.init_method}_run{self.run_id}.csv")
        filepath = os.path.join(output_dir, filename)

        if not self.entries:
            print("Nothing to save — log is empty.")
            return

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(self.entries[0]).keys())
            writer.writeheader()
            writer.writerows(asdict(e) for e in self.entries)

        print(f"Log saved to {filepath} ({len(self.entries)} entries)")
        return filepath

    def summary(self):
        if not self.entries:
            print("No entries logged.")
            return

        total = len(self.entries)
        improvements = sum(1 for e in self.entries if e.improved)
        initial_cost = self.entries[0].cost_before
        final_best = self.entries[-1].best_cost

        counts = {name: 0 for name in NEIGHBORHOOD_NAMES.values()}
        contributions = {name: 0 for name in NEIGHBORHOOD_NAMES.values()}
        for e in self.entries:
            if e.neighborhood in counts:
                counts[e.neighborhood] += 1
            if e.improved and e.neighborhood in contributions:
                contributions[e.neighborhood] += 1

        avg_shake_delta = sum(e.shake_delta for e in self.entries) / total
        avg_ls_recovery = sum(e.ls_recovery for e in self.entries) / total

        print(f"\n{'=' * 55}")
        print(f"  Instance : {self.instance_name} | k_max: {self.k_max} | "
              f"no_improve: {self.max_no_improve} | init: {self.init_method} | seed: {self.run_id}")
        print(f"{'=' * 55}")
        print(f"  Total iterations     : {total}")
        print(f"  Improvements         : {improvements} ({100 * improvements / total:.1f}%)")
        print(f"  Initial cost         : {initial_cost:.2f}")
        print(f"  Final best cost      : {final_best:.2f}")
        print(f"  Improvement          : {100 * (initial_cost - final_best) / initial_cost:.2f}%")
        gap = self._compute_gap(final_best)
        if gap is not None:
            print(f"  Gap from optimal     : {gap:.4f}%")
        print(f"  Avg shake disruption : {avg_shake_delta:.2f}")
        print(f"  Avg LS recovery      : {avg_ls_recovery:.2f}")
        print(f"\n  --- Neighborhood Usage & Contribution ---")
        for name in NEIGHBORHOOD_NAMES.values():
            used = counts[name]
            contrib = contributions[name]
            rate = (contrib / used * 100) if used > 0 else 0
            print(f"  {name:<20}: used {used:4d} | contributed {contrib:3d} improvements ({rate:.1f}%)")
        print(f"\n  Time elapsed         : {self.entries[-1].time_elapsed:.2f}s")
        print(f"{'=' * 55}\n")

    def to_summary_row(self) -> dict:
        if not self.entries:
            return {}

        final_best = self.entries[-1].best_cost
        gap = self._compute_gap(final_best)
        improvements = sum(1 for e in self.entries if e.improved)

        contributions = {name: 0 for name in NEIGHBORHOOD_NAMES.values()}
        for e in self.entries:
            if e.improved and e.neighborhood in contributions:
                contributions[e.neighborhood] += 1

        row = {
            "instance":         self.instance_name,
            "k_max":            self.k_max,
            "max_no_improve":   self.max_no_improve,
            "init_method":      self.init_method,
            "seed":             self.run_id,
            "optimal":          self.optimal,
            "best_cost":        final_best,
            "gap":              gap,
            "total_iters":      len(self.entries),
            "improvements":     improvements,
            "time":             self.entries[-1].time_elapsed,
        }
        for name in NEIGHBORHOOD_NAMES.values():
            row[f"contrib_{name}"] = contributions[name]

        return row