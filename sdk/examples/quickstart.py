"""One text/image turn against a running vLLM endpoint."""

from omniedu import OmniEdu

model = OmniEdu.from_pretrained("27B", base_url="http://127.0.0.1:8000/v1")

plain = model.chat(
    text="求解 x^2 - 5x + 6 = 0，并给出最终答案。",
    task="solve_reasoned",
)
print(plain.text)

with_image = model.chat(
    text="帮我判断学生这一步哪里错了，先给提示。",
    image="student_work.png",
    task="diagnose_and_correct",
)
print(with_image.text)

model.close()

