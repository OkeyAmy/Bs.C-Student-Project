The objectives of this study are as follows:
 
1. Design an AI-based model for predicting students' next-semester academic performance.
 
2. Identify academically at-risk students using predictive analytics.
 
3. Provide an interactive system for visualizing students' predicted academic performance.
 
4. Evaluate the effectiveness and accuracy of the proposed predictive system using students' academic data.
 
5. ~~Analyze students' performance patterns in previously attempted courses in order to identify potential areas of academic weakness in related courses for the upcoming semester.~~
   **Rescoped (2026-08-15):** the collected dataset has no course-level records — no course codes, per-course grades, or prerequisite structure, only semester-level aggregate counts (e.g. "9 registered, 8 passed"). Building this objective would mean fabricating a course/prerequisite table that corresponds to nothing real, which is a materially different and riskier thing than augmenting a small real sample. Objective 5 is out of scope for this iteration and noted as future work, contingent on collecting course-level transcript data.

   **Proposed dataset-fitting revision (PENDING — awaiting supervisor approval):**
   Analyze real prior-transition patterns and carryover signals to identify early indicators of academic weakness in the next semester, and estimate the predicted improvement per student from clearing carryovers (remediation/what-if).
   This drops the course-prerequisite requirement (no per-course data exists) and instead uses the semester-level signals we do have: prior→target transition matrix (persistence 0.712; tier-2→1 downgrade 7%, tier-3→2 downgrade 10%, tier-3→4 upgrade 6%) and carryover load (carryover ≥ 3 → 20% at-risk vs 7% below threshold; corr −0.68 with target_tier). Grouping-based analyses (per-faculty/programme weakness ranking, archetype clustering, rule extraction) were evaluated and rejected because the real data (n=61, only 5 at-risk students, 26 faculties with just 4 having ≥3 students) cannot support them. Status: **still under review.**
- Knowledge Gap 
The reviewed studies showed that different machine learning and artificial intelligence techniques have been used to predict students' academic performance. While some studies focused on identifying factors affecting performance, others concentrated on prediction accuracy or trend analysis. However, most of these studies did not develop a single system capable of predicting future academic performance while also identifying weaknesses in upcoming courses using students' performance in prerequisite courses. Furthermore, many of the existing studies focused primarily on developing prediction models without providing a platform for monitoring students' predicted academic performance for the next semester. 
This study seeks to fill this gap by developing an AI-based student academic performance analysis and prediction system that analyzes students' academic performance, predicts next-semester performance, identifies students who may be academically at risk, and analyzes prerequisite course performance to identify possible weaknesses in related upcoming courses within a web-based environment.
