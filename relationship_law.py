import re
import pandas as pd
from py_vncorenlp import VnCoreNLP

# Bước 1: Đọc file văn bản
with open("data/luat.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Bước 2: Tách câu đơn giản bằng regex
sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠ-ỹ])", text)

# === BƯỚC 3.1: Sử dụng REGEX để phát hiện viện dẫn văn bản pháp luật ===
vanban_regex = r"(Luật|Nghị định|Thông tư|Quyết định)[^\d]*(\d+)[-/](\d{4})(/[A-Z]+)?"

regex_results = []
for sentence in sentences:
    matches = re.findall(vanban_regex, sentence)
    for match in matches:
        vb_type = match[0]
        vb_num = match[1]
        vb_year = match[2]
        vb_suffix = match[3].replace("/", "") if match[3] else ""
        full_ref = f"{vb_type} {vb_num}/{vb_year}{vb_suffix}"
        regex_results.append({
            "Câu trích dẫn": sentence.strip(),
            "Văn bản được viện dẫn": full_ref,
            "Phương pháp": "Regex"
        })

# # === BƯỚC 3.2: Sử dụng VnCoreNLP để nhận diện thực thể pháp lý ===
# annotator = VnCoreNLP( annotators=[wseg,pos,ner], max_heap_size='-Xmx2g')

# vncore_results = []
# for sent in sentences:
#     output = annotator.ner(sent)
#     for token_group in output:
#         for word, label in token_group:
#             if label.startswith("B-") and "VAN_BAN_PHAP_LUAT" in label:
#                 vncore_results.append({
#                     "Câu trích dẫn": sent.strip(),
#                     "Văn bản được viện dẫn": word,
#                     "Phương pháp": "VnCoreNLP"
#                 })

# Bước 4: Tổng hợp và xuất kết quả
df_regex = pd.DataFrame(regex_results)
df_regex.to_csv("checklist_moi_quan_he_vanban.csv", index=False)
print("✅ Đã lưu checklist ra file 'checklist_moi_quan_he_vanban.csv'")
# df_vncore = pd.DataFrame(vncore_results)

# # Gộp kết quả 2 phương pháp
# df_all = pd.concat([df_regex, df_vncore], ignore_index=True)
# df_all.drop_duplicates(inplace=True)

# # Xuất ra CSV
# df_all.to_csv("checklist_moi_quan_he_vanban.csv", index=False)
# print("✅ Đã lưu checklist ra file 'checklist_moi_quan_he_vanban.csv'")
