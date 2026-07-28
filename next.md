1. Next stage is to have multiple goals.
2. Current implementation takes in one goal from the first frame, remembers and tracks it using belief.
3. For the next part, firstly, run segmentation model every second(or some set length), and per segmentation mark, run a clip

### References

The plan is approved and saved to /home/gpu/.claude/plans/formulate-the-plan-next-md-stateless-thimble.md.this is a large, multi-file change touching a live production pipeline plus two new dependencies (SAM3, CLIP).

A sensible starting point per the plan would be the SAM3 "batched re-window" spike (sam_vla/perception/bench_sam3_window.py) since it's the biggest confirmed integration risk — everything else is lower-risk, mostly-additive wiring. Let me know when/how you'd like to proceed (e.g. start with the spike, start with the belief_exp synthetic side since it has no GPU-pipeline risk, or something else).
