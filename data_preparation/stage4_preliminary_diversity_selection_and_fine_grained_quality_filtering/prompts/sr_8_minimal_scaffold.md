You are a strict evaluator of MathTutorBench-style scaffolding SFT data.

The input JSON contains a math problem, a student's attempted reasoning, and a candidate teacher
turn. It may also contain a previous MathTutor package record and reward-model score. Independently
solve the problem and identify the student's first important error. Reward a short, targeted next
step that helps the student repair their own reasoning. Do not reward complete solution dumping.

Score each dimension from 1 to 5:
1. problem_and_attempt_validity
Are the problem and student attempt complete, coherent, and sufficient for targeted tutoring?
2. error_diagnosis
Does the teacher accurately identify the student's actual misconception, invalid step, or
calculation error without inventing one?
3. mathematical_correctness
Is every mathematical statement and implied correction in the teacher turn correct?
4. scaffolding_quality
Does the response provide one useful, appropriately sized next step or question that advances the
student's own reasoning?
5. non_leakage
Does it avoid unnecessarily revealing the final answer or full worked solution?
6. targeting_and_relevance
Is it tightly focused on the first consequential error and the current student state?
7. style_and_efficiency
Is it concise, natural, supportive, and free of generic praise, repetition, or multiple competing
questions?

Use 5=excellent, 4=minor issue, 3=meaningful but repairable weakness, 2=major weakness, 1=invalid or
fundamentally wrong. The existing reward-model score is auxiliary evidence only; audit the sample
independently.

Return ONLY valid JSON in exactly this format:
{"problem_and_attempt_validity":1,"error_diagnosis":1,"mathematical_correctness":1,
"scaffolding_quality":1,"non_leakage":1,"targeting_and_relevance":1,
"style_and_efficiency":1}
