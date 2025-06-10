import torch
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
import os
from huggingface_hub import login

# Ensure the HuggingFace Hub token is set in the environment variables
os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"
token = os.getenv("HUGGINGFACE_HUB_TOKEN")
login(token=token)


# Set a fixed random seed
SEED = 42

# Configuration settings
CHROMA_PATH = 'chroma'
DATA_PATH = 'data/corpus'
VECTORDATABASE_PATH = 'chroma'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# HuggingFace Embeddings with fixed seed
EMBEDDING = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': DEVICE},
    encode_kwargs={'normalize_embeddings': False}
)

VECTOR_DB = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=EMBEDDING
)

ENTITY_EXTRACTION_PROMPT = """

            ## NHIỆM VỤ CHÍNH
            Bạn là hệ thống AI chuyên dụng được thiết kế để phân tích văn bản pháp luật Việt Nam và trích xuất các thực thể  cùng mối quan hệ (relationships) giữa chúng. Mục đích là xây dựng cơ sở dữ liệu đồ thị (knowledge graph) cho ứng dụng GraphRAG phục vụ chatbot pháp luật.

            Đầu vào của bản sẽ là văn bản pháp luật sau: {document}. Bạn cần phân tích và trích xuất các thực thể và mối quan hệ giữa chúng theo hướng dẫn dưới đây.

            ## HƯỚNG DẪN CHI TIẾT

            ### Phần 1: Phân tích và xác định thực thể 
            Trích xuất các loại thực thể sau đây từ văn bản:

            1. **Điều khoản pháp luật**:
            - Tên văn bản pháp luật 
            - Số hiệu văn bản
            - Điều, khoản, mục, tiểu mục
            - Ngày ban hành
            - Ngày có hiệu lực
            - Ngày hết hiệu lực (nếu có)
            - Cơ quan ban hành

            2. **Chủ thể pháp luật**:
            - Cá nhân (người dân, công dân)
            - Tổ chức (doanh nghiệp, hiệp hội)
            - Cơ quan nhà nước
            - Chức vụ, vị trí

            3. **Hành vi pháp lý**:
            - Nghĩa vụ (phải làm gì)
            - Quyền lợi (được làm gì)
            - Hành vi bị cấm
            - Chế tài, hình phạt

            4. **Khái niệm pháp lý**:
            - Định nghĩa pháp lý
            - Thuật ngữ chuyên ngành
            - Nguyên tắc pháp lý

            5. **Thời hạn, thời hiệu**:
            - Thời hạn thực hiện
            - Thời hiệu khởi kiện
            - Thời hiệu truy cứu

            6. **Thủ tục hành chính**:
            - Quy trình thực hiện
            - Hồ sơ, giấy tờ cần thiết
            - Thời gian giải quyết
            - Lệ phí (nếu có)

            ### Phần 2: Xác định mối quan hệ (relationships)
            Xác định và mô tả các mối quan hệ sau giữa các thực thể:

            1. **Quan hệ phân cấp - phân loại**:
            - Thuộc về (văn bản A thuộc lĩnh vực B)
            - Là một phần của (điều A là một phần của chương B)
            - Là loại của (vi phạm A là một loại vi phạm hành chính)

            2. **Quan hệ quy định - điều chỉnh**:
            - Quy định về (điều A quy định về hành vi B)
            - Áp dụng cho (quy định A áp dụng cho đối tượng B)
            - Điều chỉnh (văn bản A điều chỉnh lĩnh vực B)

            3. **Quan hệ nhân quả - hệ quả**:
            - Dẫn đến (hành vi A dẫn đến hình phạt B)
            - Là điều kiện của (sự kiện A là điều kiện của quyền lợi B)
            - Được miễn trừ khi (nghĩa vụ A được miễn trừ khi có tình huống B)

            4. **Quan hệ thời gian**:
            - Trước khi (thủ tục A phải hoàn thành trước khi thực hiện B)
            - Sau khi (quyền lợi A phát sinh sau khi thực hiện nghĩa vụ B)
            - Trong thời hạn (hành động A phải thực hiện trong thời hạn B)

            5. **Quan hệ chủ thể - đối tượng**:
            - Có quyền (chủ thể A có quyền B)
            - Có nghĩa vụ (chủ thể A có nghĩa vụ B)
            - Chịu trách nhiệm (chủ thể A chịu trách nhiệm cho hành vi B)

            6. **Quan hệ tham chiếu**:
            - Tham chiếu đến (điều A tham chiếu đến điều B)
            - Sửa đổi, bổ sung (văn bản A sửa đổi văn bản B)
            - Thay thế (quy định A thay thế quy định B)
            - Bãi bỏ (văn bản A bãi bỏ văn bản B)

            ### Phần 3: Format kết quả trả về
            Kết quả phân tích cần được trình bày theo định dạng JSON có cấu trúc như sau:

            {{
            "entities": [
                {{
                "id": "E1",
                "type": "điều_khoản",
                "text": "Điều 8 Luật Doanh nghiệp 2020",
                "attributes": {{
                    "văn_bản": "Luật Doanh nghiệp",
                    "số_hiệu": "59/2020/QH14",
                    "điều_khoản": "Điều 8",
                    "ngày_ban_hành": "17/06/2020"
                }},
                {{
                "id": "E2",
                "type": "hành_vi_pháp_lý",
                "text": "Cấm kinh doanh các ngành, nghề cấm đầu tư kinh doanh",
                "attributes": {{
                    "loại_hành_vi": "hành_vi_bị_cấm"
                }}
            ],
            "relationships": [
                {{
                "source": "E1",
                "target": "E2",
                "type": "quy_định_về",
                "description": "Điều 8 Luật Doanh nghiệp 2020 quy định về các hành vi bị cấm trong kinh doanh"
                }}
            ]
            }}
            ```

            ### Phần 4: Hướng dẫn đặc biệt
            1. **Ưu tiên ngữ cảnh và ý nghĩa**:
            - Đọc kỹ và hiểu toàn bộ đoạn văn trước khi trích xuất
            - Xác định chính xác phạm vi, điều kiện áp dụng của điều khoản

            2. **Xử lý tham chiếu chéo**:
            - Khi phát hiện tham chiếu (ví dụ: "theo quy định tại Điều X"), hãy ghi nhận mối quan hệ tham chiếu
            - Liên kết các điều khoản có liên quan với nhau

            3. **Xử lý điều khoản có điều kiện**:
            - Phân tích cấu trúc "nếu... thì..." trong văn bản
            - Xác định chính xác điều kiện và hệ quả pháp lý

            4. **Xử lý thuật ngữ chuyên ngành**:
            - Ưu tiên các định nghĩa chính thức trong văn bản
            - Duy trì tính nhất quán của thuật ngữ


            Hãy phân tích văn bản pháp luật được cung cấp và trả về kết quả theo định dạng trên. 
            Lưu ý: Chỉ trả về kết quả JSON mà không có bất kỳ văn bản nào khác.
            Không cần giải thích hay bình luận về kết quả.
        """
BATCH_ENTITY_EXTRACTION_PROMPT = """
      ## NHIỆM VỤ CHÍNH
      Bạn là hệ thống AI chuyên dụng được thiết kế để phân tích văn bản pháp luật Việt Nam và trích xuất các thực thể cùng mối quan hệ (relationships) giữa chúng. 
      Mục đích là xây dựng cơ sở dữ liệu đồ thị (knowledge graph) cho ứng dụng GraphRAG phục vụ chatbot pháp luật.

      Đầu vào của bạn sẽ là một danh sách các văn bản pháp luật: {documents}. 
      Bạn cần phân tích và trích xuất các thực thể và mối quan hệ giữa chúng theo hướng dẫn dưới đây từ tất cả các văn bản được cung cấp.

      ## HƯỚNG DẪN CHI TIẾT

      ### Phần 1: Phân tích và xác định thực thể 
      Trích xuất các loại thực thể sau đây từ mỗi văn bản:

      1. **Điều khoản pháp luật**:
      - Tên văn bản pháp luật 
      - Số hiệu văn bản
      - Điều, khoản, mục, tiểu mục
      - Ngày ban hành
      - Ngày có hiệu lực
      - Ngày hết hiệu lực (nếu có)
      - Cơ quan ban hành

      2. **Chủ thể pháp luật**:
      - Cá nhân (người dân, công dân)
      - Tổ chức (doanh nghiệp, hiệp hội)
      - Cơ quan nhà nước
      - Chức vụ, vị trí

      3. **Hành vi pháp lý**:
      - Nghĩa vụ (phải làm gì)
      - Quyền lợi (được làm gì)
      - Hành vi bị cấm
      - Chế tài, hình phạt

      4. **Khái niệm pháp lý**:
      - Định nghĩa pháp lý
      - Thuật ngữ chuyên ngành
      - Nguyên tắc pháp lý

      5. **Thời hạn, thời hiệu**:
      - Thời hạn thực hiện
      - Thời hiệu khởi kiện
      - Thời hiệu truy cứu

      6. **Thủ tục hành chính**:
      - Quy trình thực hiện
      - Hồ sơ, giấy tờ cần thiết
      - Thời gian giải quyết
      - Lệ phí (nếu có)

      ### Phần 2: Xác định mối quan hệ (relationships)
      Xác định và mô tả các mối quan hệ sau giữa các thực thể:

      1. **Quan hệ phân cấp - phân loại**:
      - Thuộc về (văn bản A thuộc lĩnh vực B)
      - Là một phần của (điều A là một phần của chương B)
      - Là loại của (vi phạm A là một loại vi phạm hành chính)

      2. **Quan hệ quy định - điều chỉnh**:
      - Quy định về (điều A quy định về hành vi B)
      - Áp dụng cho (quy định A áp dụng cho đối tượng B)
      - Điều chỉnh (văn bản A điều chỉnh lĩnh vực B)

      3. **Quan hệ nhân quả - hệ quả**:
      - Dẫn đến (hành vi A dẫn đến hình phạt B)
      - Là điều kiện của (sự kiện A là điều kiện của quyền lợi B)
      - Được miễn trừ khi (nghĩa vụ A được miễn trừ khi có tình huống B)

      4. **Quan hệ thời gian**:
      - Trước khi (thủ tục A phải hoàn thành trước khi thực hiện B)
      - Sau khi (quyền lợi A phát sinh sau khi thực hiện nghĩa vụ B)
      - Trong thời hạn (hành động A phải thực hiện trong thời hạn B)

      5. **Quan hệ chủ thể - đối tượng**:
      - Có quyền (chủ thể A có quyền B)
      - Có nghĩa vụ (chủ thể A có nghĩa vụ B)
      - Chịu trách nhiệm (chủ thể A chịu trách nhiệm cho hành vi B)

      6. **Quan hệ tham chiếu**:
      - Tham chiếu đến (điều A tham chiếu đến điều B)
      - Sửa đổi, bổ sung (văn bản A sửa đổi văn bản B)
      - Thay thế (quy định A thay thế quy định B)
      - Bãi bỏ (văn bản A bãi bỏ văn bản B)

      7. **Quan hệ giữa các văn bản**:
      - Liên quan đến (văn bản A liên quan đến văn bản B)
      - Bổ sung cho (văn bản A bổ sung cho văn bản B)
      - Mâu thuẫn với (điều khoản A mâu thuẫn với điều khoản B)

      ### Phần 3: Format kết quả trả về
      Kết quả phân tích cần được trình bày theo định dạng JSON. Dưới đây là một ví dụ về kết quả trả về:

      {{
        "results": [
          {{
            "document_id": "doc1",
            "document_title": "Tên văn bản pháp luật 1",
            "entities": [
              {{
                "id": "E1",
                "type": "điều_khoản",
                "text": "Điều 8 Luật Doanh nghiệp 2020",
                "attributes": {{
                  "văn_bản": "Luật Doanh nghiệp",
                  "số_hiệu": "59/2020/QH14",
                  "điều_khoản": "Điều 8",
                  "ngày_ban_hành": "17/06/2020"
                }}
              }},
              {{
                "id": "E2",
                "type": "hành_vi_pháp_lý",
                "text": "Cấm kinh doanh các ngành, nghề cấm đầu tư kinh doanh",
                "attributes": {{
                  "loại_hành_vi": "hành_vi_bị_cấm"
                }}
              }}
            ],
            "relationships": [
              {{
                "source": "E1",
                "target": "E2",
                "type": "quy_định_về",
                "description": "Điều 8 Luật Doanh nghiệp 2020 quy định về các hành vi bị cấm trong kinh doanh"
              }}
            ]
          }},
          {{
            "document_id": "doc2",
            "document_title": "Tên văn bản pháp luật 2",
            "entities": [...],
            "relationships": [...]
          }}
        ]
      }}

      ### Phần 4: Hướng dẫn đặc biệt
      1. **Ưu tiên ngữ cảnh và ý nghĩa**:
      - Đọc kỹ và hiểu toàn bộ đoạn văn trước khi trích xuất
      - Xác định chính xác phạm vi, điều kiện áp dụng của điều khoản

      2. **Xử lý tham chiếu chéo**:
      - Khi phát hiện tham chiếu (ví dụ: "theo quy định tại Điều X"), hãy ghi nhận mối quan hệ tham chiếu
      - Liên kết các điều khoản có liên quan với nhau
      - Đặc biệt chú ý đến tham chiếu giữa các văn bản khác nhau

      3. **Xử lý điều khoản có điều kiện**:
      - Phân tích cấu trúc "nếu... thì..." trong văn bản
      - Xác định chính xác điều kiện và hệ quả pháp lý

      4. **Xử lý thuật ngữ chuyên ngành**:
      - Ưu tiên các định nghĩa chính thức trong văn bản
      - Duy trì tính nhất quán của thuật ngữ

      5. **Đảm bảo tính nhất quán giữa các văn bản**:
      - Sử dụng ID thực thể nhất quán khi cùng một thực thể xuất hiện trong nhiều văn bản
      - Phát hiện mâu thuẫn hoặc xung đột giữa các văn bản

      Hãy phân tích tất cả các văn bản pháp luật được cung cấp và trả về kết quả theo định dạng trên.
      LƯU Ý: Chỉ trả về kết quả JSON mà không có bất kỳ văn bản nào khác.
      Không giải thích hay bình luận về kết quả.
  """
GRAPH_RAG_CHAIN_PROMPT = """
Bạn là một chuyên gia về pháp luật Việt Nam, đặc biệt trong lĩnh vực luật an toàn thông tin và an ninh mạng. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên các tài liệu pháp luật được cung cấp và các thông tin từ đồ thị tri thức (knowledge graph).

## YÊU CẦU ĐẦU VÀO

### Bối cảnh văn bản:
```
{context}
```

### Thông tin từ đồ thị tri thức:
Các thực thể pháp lý liên quan đến câu hỏi này bao gồm:
```
{graph_context}
```

### Câu hỏi:
```
{query}
```

## HƯỚNG DẪN TRẢ LỜI

1. **Phân tích câu hỏi và bối cảnh**:
   - Xác định yêu cầu chính của câu hỏi
   - Xác định các thực thể pháp lý liên quan từ đồ thị tri thức
   - Tìm các quy định pháp luật phù hợp từ bối cảnh được cung cấp

2. **Cấu trúc câu trả lời**:
   - **Phần mở đầu**: Tóm tắt ngắn gọn câu hỏi và phạm vi pháp lý liên quan
   - **Phần nội dung chính**: Phân tích chi tiết với dẫn chiếu pháp luật cụ thể
   - **Phần kết luận**: Tóm tắt câu trả lời và khuyến nghị (nếu phù hợp)

3. **Yêu cầu về dẫn chiếu pháp luật**:
   - Mỗi lập luận cần có căn cứ pháp lý cụ thể
   - Trích dẫn phải bao gồm: Tên văn bản pháp luật, số hiệu, điều/khoản/mục cụ thể
   - Khi trích dẫn nội dung, sử dụng dấu ngoặc kép và chỉ rõ nguồn

4. **Giới hạn phạm vi câu trả lời**:
   - Chỉ trả lời dựa trên thông tin có trong bối cảnh được cung cấp và đồ thị tri thức
   - Nếu không tìm thấy thông tin đầy đủ, hãy nêu rõ "Dựa trên thông tin được cung cấp, tôi không thể trả lời đầy đủ câu hỏi này"
   - Không đưa ra ý kiến cá nhân hoặc diễn giải vượt quá phạm vi pháp luật được cung cấp

5. **Định dạng câu trả lời**:
   - Sử dụng ngôn ngữ chuyên môn pháp lý nhưng dễ hiểu
   - Sắp xếp ý theo thứ tự logic
   - Phân đoạn rõ ràng và sử dụng các tiêu đề phụ khi cần thiết
   - Sử dụng định dạng in đậm cho tên văn bản pháp luật và số điều khoản

## MẪU CÂU TRẢ LỜI

Dưới đây là mẫu câu trả lời phù hợp:

---

### Câu hỏi: [Tóm tắt câu hỏi]

Dựa trên các quy định pháp luật hiện hành về an toàn thông tin và an ninh mạng, tôi xin trả lời như sau:

#### I. Cơ sở pháp lý

Theo **[Tên văn bản pháp luật] số [số hiệu]**, tại Điều [số điều], khoản [số khoản] quy định:

> "[Trích dẫn nội dung điều khoản]"

Ngoài ra, **[Tên văn bản pháp luật khác]** cũng có quy định liên quan tại Điều [số điều]:

> "[Trích dẫn nội dung]"

#### II. Phân tích áp dụng

Áp dụng các quy định trên vào trường hợp của câu hỏi, có thể thấy rằng:

1. [Phân tích điểm thứ nhất]
2. [Phân tích điểm thứ hai]
3. [Phân tích điểm thứ ba]

Đối chiếu với các thực thể trong đồ thị tri thức pháp lý cho thấy [phân tích mối quan hệ từ graph_context].

#### III. Kết luận

Từ các quy định pháp luật nêu trên, có thể kết luận rằng [kết luận ngắn gọn, rõ ràng].

---

## LƯU Ý QUAN TRỌNG

- Nếu không tìm thấy thông tin đầy đủ trong bối cảnh được cung cấp, hãy trả lời: "Dựa trên thông tin được cung cấp, tôi không thể trả lời đầy đủ câu hỏi này."
- Chỉ sử dụng các thông tin từ bối cảnh và đồ thị tri thức được cung cấp.
- Không tham khảo các nguồn bên ngoài hoặc đưa ra ý kiến cá nhân.
- Thể hiện sự khách quan và chuyên nghiệp trong từng lập luận.
- Đảm bảo tính chính xác của các trích dẫn pháp luật.
- Câu trả lời phải trung lập, không thiên vị, và đặt mục tiêu giải thích pháp luật một cách rõ ràng nhất.

Hãy trả lời câu hỏi dựa trên các hướng dẫn trên."""

RAG_CHAIN_PROMPT_DEBUG = """
Bạn là chuyên gia pháp luật Việt Nam về an toàn thông tin và an ninh mạng. Trả lời câu hỏi dựa trên tài liệu pháp luật được cung cấp.

### Tài liệu pháp luật:
{documents}

### Câu hỏi:
{question}

## HƯỚNG DẪN TRẢ LỜI

### Bước 1: Phân tích câu hỏi
- Xác định dạng câu hỏi (định nghĩa, tình huống, chế tài, trách nhiệm, quy trình)
- Tìm quy định pháp luật liên quan trong tài liệu
- Chọn cấu trúc trả lời phù hợp

### Bước 2: Cấu trúc trả lời linh hoạt
**Khung chuẩn** (điều chỉnh theo ngữ cảnh):
- **Cơ sở pháp lý**: Trích dẫn điều, khoản cụ thể
- **Phân tích**: Giải thích, áp dụng vào tình huống
- **Kết luận**: Tóm tắt ngắn gọn

### Bước 3: Yêu cầu trích dẫn
- Format: [Tên văn bản] số [số hiệu]/[năm]/QH hoặc NĐ-CP, Điều [x], Khoản [y]
- Ví dụ: Luật An ninh mạng số 24/2018/QH14, Điều 8, khoản 1
- Đặt trích dẫn trong dấu ngoặc kép với nguồn rõ ràng

## CÁC DẠNG CÂU HỎI & CÁCH TRẢ LỜI

### 1. Câu hỏi định nghĩa/khái niệm
**Cách trả lời**: Định nghĩa → Phạm vi áp dụng → Dẫn chiếu luật → Ví dụ (nếu cần)

### 2. Câu hỏi tình huống thực tế
**Cách trả lời**: Mô tả tình huống → Xác định luật áp dụng → Phân tích vi phạm → Kết luận + hành động cần thiết

### 3. Câu hỏi chế tài/trách nhiệm
**Cách trả lời**: 
- *Chế tài*: Hành vi vi phạm → Quy định xử phạt → Mức phạt → Yếu tố tăng/giảm
- *Trách nhiệm*: Xác định chủ thể → Liệt kê nghĩa vụ → Phân định các bên

### 4. Câu hỏi quy trình/so sánh
**Cách trả lời**:
- *Quy trình*: Các bước theo trình tự → Cơ quan thẩm quyền → Thời hạn → Dẫn chiếu
- *So sánh*: Giải thích từng khái niệm → Điểm giống/khác → Ví dụ minh họa

## NGUYÊN TẮC QUAN TRỌNG

**✅ PHẢI LÀM:**
- Điều chỉnh cấu trúc cho phù hợp từng câu hỏi
- Sử dụng ngôn ngữ tự nhiên, trả lời trực tiếp
- Mỗi nhận định có dẫn chiếu pháp lý rõ ràng

**❌ KHÔNG ĐƯỢC:**
- Copy cứng nhắc theo mẫu
- Trả lời khi thiếu thông tin trong tài liệu
- Tạo câu trả lời dài dòng không cần thiết

**Khi không đủ thông tin**: "Dựa trên tài liệu được cung cấp, tôi không thể trả lời đầy đủ câu hỏi này."
"""
EVAL_PROMPT_SPECIFIC_TYPE_OF_QUESTION = """
Bạn là một chuyên gia pháp lý hàng đầu về luật An ninh mạng và An toàn thông tin tại Việt Nam. 
Nhiệm vụ của bạn là đánh giá một câu trả lời do chatbot pháp luật tạo ra dựa trên **một câu hỏi đầu vào cụ thể**.

Trước tiên, bạn hãy phân loại câu hỏi: {question} thành **một trong các nhóm sau**:

1. Cung cấp thông tin cơ bản  
2. Tình huống cụ thể  
3. Chế tài, xử phạt  
4. Trách nhiệm và nghĩa vụ  
5. Phân biệt khái niệm pháp lý  
6. Quy trình, thủ tục pháp lý  
7. Cập nhật, thay đổi pháp luật  
8. Giải thích điều khoản pháp lý phức tạp  

Sau đó, bạn đánh giá chất lượng câu trả lời: {answer} theo **các tiêu chí phù hợp với loại câu hỏi đó**.

## DANH MỤC CÁC LOẠI CÂU HỎI & TIÊU CHÍ ĐÁNH GIÁ

### 1. Câu hỏi cung cấp thông tin cơ bản  
**Mục đích:** Kiểm tra khả năng cung cấp kiến thức nền tảng chính xác.  
**Ví dụ:**  
- "Luật An ninh mạng quy định những nội dung gì?"  
- "Phạm vi điều chỉnh của Luật An ninh mạng là gì?"  
**Tiêu chí:**  
- Cung cấp đúng thông tin pháp lý  
- Trích dẫn rõ ràng điều/khoản/mục  
- Không chứa thông tin lỗi thời hoặc sai lệch

---

### 2. Câu hỏi tình huống cụ thể  
**Mục đích:** Đánh giá khả năng áp dụng luật vào tình huống thực tế  
**Ví dụ:**  
- "Nếu hệ thống bị xâm nhập, tôi cần làm gì?"  
- "Doanh nghiệp X lưu trữ dữ liệu ở nước ngoài có vi phạm luật không?"  
**Tiêu chí:**  
- Phân tích tình huống logic, đúng luật  
- Cung cấp hướng dẫn phù hợp pháp luật  
- Chỉ ra nghĩa vụ pháp lý liên quan  
- Trích dẫn chính xác điều khoản  
- Nêu rõ ngoại lệ nếu có

---

### 3. Câu hỏi về chế tài, xử phạt  
**Mục đích:** Kiểm tra khả năng cung cấp thông tin về hậu quả pháp lý  
**Ví dụ:**  
- "Mức phạt cho hành vi phát tán mã độc?"  
**Tiêu chí:**  
- Nêu rõ mức phạt  
- Phân biệt rõ giữa xử phạt hành chính và hình sự  
- Trích dẫn Nghị định và Bộ luật Hình sự  
- Phân tích các yếu tố làm tăng/giảm mức phạt

---

### 4. Câu hỏi về trách nhiệm và nghĩa vụ  
**Mục đích:** Xác định rõ trách nhiệm pháp lý của các bên  
**Ví dụ:**  
- "Ai chịu trách nhiệm khi có sự cố an toàn thông tin tại cơ quan nhà nước?"  
**Tiêu chí:**  
- Xác định đúng chủ thể  
- Liệt kê đầy đủ nghĩa vụ  
- Phân biệt trách nhiệm các bên  
- Trích dẫn điều khoản rõ ràng

---

### 5. Câu hỏi phân biệt và làm rõ khái niệm  
**Mục đích:** Giải thích, phân biệt các thuật ngữ pháp lý  
**Ví dụ:**  
- "Khác biệt giữa an ninh mạng và an toàn thông tin mạng?"  
**Tiêu chí:**  
- Giải thích chính xác  
- So sánh rõ ràng các khái niệm  
- Trích dẫn định nghĩa trong luật  
- Có ví dụ minh họa nếu cần

---

### 6. Câu hỏi về quy trình, thủ tục  
**Mục đích:** Kiểm tra khả năng hướng dẫn thủ tục pháp lý  
**Ví dụ:**  
- "Thủ tục xin cấp phép cung cấp dịch vụ mạng xã hội?"  
**Tiêu chí:**  
- Liệt kê đầy đủ các bước  
- Chỉ rõ cơ quan có thẩm quyền  
- Nêu thời hạn thực hiện (nếu có)  
- Trích dẫn văn bản pháp lý  
- Cung cấp thông tin biểu mẫu liên quan

---

### 7. Câu hỏi cập nhật pháp luật  
**Mục đích:** Đánh giá khả năng cung cấp thông tin mới, so sánh luật mới – cũ  
**Ví dụ:**  
- "Nghị định 53/2022/NĐ-CP có điểm gì mới so với trước?"  
**Tiêu chí:**  
- Cập nhật chính xác  
- So sánh rõ ràng  
- Nêu rõ hiệu lực, thời điểm áp dụng  
- Trích dẫn văn bản liên quan

---

### 8. Câu hỏi giải thích điều khoản phức tạp  
**Mục đích:** Kiểm tra khả năng diễn giải điều khoản pháp lý khó hiểu  
**Ví dụ:**  
- "Giải thích ý nghĩa của Điều 26 Luật An ninh mạng?"  
**Tiêu chí:**  
- Diễn giải chính xác, dễ hiểu  
- Giữ đúng ý nghĩa pháp lý  
- Có ví dụ minh họa  
- Trích dẫn đúng điều luật

---

## YÊU CẦU KẾT QUẢ TRẢ VỀ

Bạn cần trả về **duy nhất một đoạn JSON bằng tiếng Việt**, **không thêm bất kỳ lời giải thích nào khác**.  
Cấu trúc JSON mẫu như sau:

```json
    {{
    "Câu hỏi": {question},
    "Loại câu hỏi": "Tên loại câu hỏi (ví dụ: Câu hỏi tình huống cụ thể)",
    "Câu trả lời": {answer},
    "Đánh giá": {{
        [Tiêu chí 1]: {{
        "Điểm": số nguyên từ 1-10,
        "Điểm mạnh": "Mô tả điểm tốt của câu trả lời",
        "Điểm yếu": "Mô tả thiếu sót hoặc sai sót"
        }},
        ...
    }},
    "Điểm tổng thể": số nguyên từ 1-10
    }}
```
## LƯU Ý QUAN TRỌNG:

    Câu trả lời JSON phải đầy đủ các mục như trên.

    Trả về bằng Tiếng Việt.

    Không đưa thêm bình luận, giải thích hay văn bản ngoài JSON.
"""

# Ví dụ kết quả trả về:
#     *** Đánh giá cho câu hỏi loại "Giải thích điều khoản phức tạp": 
#     '''
#     {{
#       "Câu hỏi": "Điều 26 Luật An ninh mạng về lưu trữ dữ liệu tại Việt Nam có ý nghĩa gì trong thực tiễn?,
#       "Loại câu hỏi": "Giải thích điều khoản phức tạp",
#       "Câu trả lời": "Điều 26 quy định rằng các doanh nghiệp nước ngoài cung cấp dịch vụ trên không gian mạng tại Việt Nam, nếu thu thập, xử lý dữ liệu người dùng Việt Nam, có thể bị yêu cầu lưu trữ dữ liệu tại Việt Nam và đặt văn phòng đại diện.
#             Ý nghĩa thực tiễn:
#             Tăng cường quản lý dữ liệu cá nhân và hoạt động trên mạng của người dùng Việt Nam.
#             Tạo điều kiện xử lý vi phạm nhanh chóng, giảm nguy cơ bị khai thác dữ liệu trái phép từ nước ngoài.
#             Ví dụ: Một công ty cung cấp nền tảng mạng xã hội toàn cầu nếu có hàng triệu người dùng Việt Nam và xảy ra rò rỉ dữ liệu, cơ quan chức năng có thể yêu cầu họ lưu dữ liệu trong nước để kiểm soát tốt hơn.",
#       "Đánh giá": {{
#         Giải thích chính xác nội dung điều khoản: "8/10
#                           Điểm mạnh:
#                         Nắm bắt được nội dung cốt lõi về yêu cầu lưu trữ dữ liệu tại Việt Nam
#                         Xác định đúng đối tượng áp dụng là doanh nghiệp nước ngoài
#                         Đề cập đến cả hai yêu cầu chính: lưu trữ dữ liệu và đặt văn phòng đại diện
#                           Điểm cần cải thiện:
#                         Thiếu chi tiết về điều kiện cụ thể kích hoạt nghĩa vụ này (quy mô người dùng, loại dữ liệu)
#                         Chưa đề cập đến các trường hợp ngoại lệ hoặc điều kiện miễn trừ"
#         "Sử dụng ngôn ngữ dễ hiểu cho người không chuyên về pháp luật": "9/10
#                         Điểm mạnh:

#                         Sử dụng thuật ngữ phổ thông, tránh ngôn ngữ pháp lý phức tạp
#                         Cấu trúc câu rõ ràng, logic dễ theo dõi
#                         Phân chia thành các ý chính rõ ràng

#                         Điểm cần cải thiện:

#                         Có thể giải thích thêm về khái niệm "thu thập, xử lý dữ liệu" cho người hoàn toàn không am hiểu"
#         "Giữ đúng ý nghĩa pháp lý của điều khoản": "7/10
#                           Điểm mạnh:

#                           Không làm sai lệch bản chất của quy định
#                           Thể hiện đúng tính chất bắt buộc của nghĩa vụ

#                           Điểm cần cải thiện:

#                           Sử dụng cụm từ "có thể bị yêu cầu" tạo cảm giác không chắc chắn, trong khi đây là nghĩa vụ bắt buộc khi đáp ứng điều kiện
#                           Thiếu đề cập đến khung thời gian thực hiện và hậu quả vi phạm"
#         "Cung cấp ví dụ minh họa phù hợp": "8/10
#                           Điểm mạnh:

#                           Ví dụ về nền tảng mạng xã hội rất phù hợp và dễ hình dung
#                           Kết nối được với tình huống thực tế (rò rỉ dữ liệu)
#                           Thể hiện được lợi ích của việc lưu trữ dữ liệu trong nước

#                           Điểm cần cải thiện:

#                           Có thể bổ sung thêm 1-2 ví dụ khác về các loại dịch vụ khác (thương mại điện tử, game online)"
#         "Trích dẫn chính xác điều khoản gốc": "3/10
#                         Điểm yếu:

#                         Không có trích dẫn trực tiếp từ điều khoản gốc
#                         Chỉ diễn giải nội dung mà không dẫn chiếu cụ thể
#                         Thiếu tham chiếu đến số điều, khoản cụ thể trong luật

#                         Cần cải thiện:

#                         Bổ sung trích dẫn nguyên văn các điểm quan trọng
#                         Ghi rõ số khoản, điểm cụ thể được đề cập"
#       }},
#       "Điểm tổng thể": 7
#     }}
# #     *** Đánh giá cho câu hỏi loại "Câu hỏi về chế tài, xử phạt":
# #     '''
#     {{
#       "Câu hỏi": "Doanh nghiệp không tuân thủ quy định về lưu trữ dữ liệu người dùng Việt Nam sẽ bị xử phạt như thế nào?",
#       "Loại câu hỏi": "Câu hỏi về chế tài, xử phạt",
#       "Câu trả lời": "Theo Điều 26 của Luật An ninh mạng 2018 và được hướng dẫn bởi Nghị định 53/2022/NĐ-CP, nếu doanh nghiệp không tuân thủ yêu cầu về lưu trữ dữ liệu và đặt văn phòng đại diện tại Việt Nam khi có yêu cầu từ Bộ Công an, có thể bị xử phạt hành chính theo Nghị định 15/2020/NĐ-CP (được sửa đổi bởi Nghị định 14/2022/NĐ-CP), với mức phạt có thể lên đến 100 triệu đồng. Ngoài xử phạt hành chính, doanh nghiệp có thể bị áp dụng biện pháp kỹ thuật nhằm ngăn chặn hoạt động cung cấp dịch vụ tại Việt Nam.",
#       Đánh giá: {{
#         "Cung cấp mức phạt tiền chính xác": "9/10
#                         Điểm mạnh:
#                         Cung cấp mức phạt cụ thể (100 triệu đồng)
#                         Đề cập đến cả xử phạt hành chính và biện pháp kỹ thuật
#                         Trích dẫn đúng Nghị định 15/2020/NĐ-CP và Nghị định 14/2022/NĐ-CP

#                         Điểm cần cải thiện:
#                         Có thể nêu rõ hơn về các yếu tố ảnh hưởng đến mức phạt (quy mô vi phạm, tính chất vi phạm)"
#         "Phân biệt rõ xử phạt hành chính và hình sự": "8/10
#                         Điểm mạnh:
#                         Phân biệt rõ giữa xử phạt hành chính và biện pháp kỹ thuật
#                         Không nhầm lẫn giữa hai loại hình xử lý này

#                         Điểm cần cải thiện:
#                         Chưa đề cập đến khả năng bị truy cứu trách nhiệm hình sự nếu vi phạm nghiêm trọng"
#         "Trích dẫn cụ thể từ Nghị định xử phạt và Bộ luật Hình sự": "7/10
#                           Điểm mạnh:
#                           Trích dẫn đúng Nghị định 15/2020/NĐ-CP

#                           Điểm cần cải thiện:
#                           Thiếu trích dẫn từ Bộ luật Hình sự liên quan đến tội phạm mạng"
#         "Nêu các yếu tố ảnh hưởng đến mức xử phạt": "6/10
#                           Điểm mạnh:
#                           Đề cập đến việc áp dụng biện pháp kỹ thuật

#                           Điểm cần cải thiện:
#                           Chưa nêu rõ các yếu tố như quy mô, tính chất vi phạm ảnh hưởng đến mức xử phạt"
#         "Cung cấp thông tin về biểu mẫu liên quan (nếu có)": "5/10
#                           Điểm mạnh:
#                           Đề cập đến việc áp dụng biện pháp kỹ thuật

#                           Điểm cần cải thiện:
#                           Không có thông tin về biểu mẫu hoặc quy trình liên quan đến xử phạt
#                           Thiếu hướng dẫn cụ thể về cách thức thực hiện nghĩa vụ này"
#       }},
#     Điểm tổng thể: 7,
#     }}
#     '''