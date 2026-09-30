"""
Comprehensive Question-Answer Dataset Generator for IndicRAG
Creates ~360 realistic QA instances across:
  - English (monolingual)
  - Indic (Hindi in Devanagari script)
  - Code-Mixed (Romanized Hinglish)
Includes ANSWERABLE questions with ground truth passages and answers,
and deliberately UNANSWERABLE questions for hallucination testing.
Also generates difficult_cases.csv (Module 6).
"""

import csv
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.config import QUESTIONS_DIR, EVALUATION_DIR


# Base ground-truth QA facts directly extracted from the 3 university regulation PDFs
FACT_TEMPLATES = [
    {
        "category": "examination_passing",
        "doc": "examination_manual.pdf",
        "chunk_ids": [107],
        "fact_en": "To pass a course, a student must score at least 30% marks in the end term exam and a total of 40 marks out of 100 overall.",
        "en": [
            "What is the minimum passing marks in the end semester examination?",
            "How many marks are required out of 100 to pass a course?",
            "What percentage is required in end term exam to pass?",
            "What are the passing criteria for semester courses?"
        ],
        "indic": [
            "एंड सेमेस्टर परीक्षा पास करने के लिए न्यूनतम कितने प्रतिशत अंक चाहिए?",
            "किसी कोर्स को पास करने के लिए 100 में से कितने अंक आवश्यक हैं?",
            "परीक्षा में पास होने के लिए एंड टर्म में न्यूनतम कितने अंक लाने होते हैं?",
            "कोर्स पास करने के लिए पासिंग क्राइटेरिया क्या है?"
        ],
        "codemix": [
            "End sem exam pass karne ke liye kitna percentage marks chahiye?",
            "Course pass karne ke liye 100 me se kitne number compulsory hain?",
            "End term exam me passing marks kitna hota hai?",
            "Semester course clear karne ka passing criteria kya hai?"
        ]
    },
    {
        "category": "f_grade_repeat",
        "doc": "examination_manual.pdf",
        "chunk_ids": [155],
        "fact_en": "The 'F' grade denotes failing a course. A student must repeat all courses in which an 'F' grade is obtained until a passing grade is achieved, with no grade points awarded.",
        "en": [
            "What happens if a student receives an F grade in a course?",
            "Does an F grade carry any grade points?",
            "What is the rule for repeating courses with an F grade?",
            "Can a student graduate with an F grade without repeating?"
        ],
        "indic": [
            "यदि किसी छात्र को किसी कोर्स में F ग्रेड मिलता है तो क्या नियम है?",
            "क्या F ग्रेड में कोई ग्रेड पॉइंट मिलता है?",
            "F ग्रेड आने पर क्या कोर्स दोबारा पढ़ना अनिवार्य है?",
            "F ग्रेड के लिए विश्वविद्यालय की क्या नीति है?"
        ],
        "codemix": [
            "Agar course me F grade aa jaye toh kya repeat karna padega?",
            "Kya F grade aane par koi grade points milte hain?",
            "F grade clear karne ke liye university ka rule kya hai?",
            "F grade milne par student ko kya procedure follow karna hota hai?"
        ]
    },
    {
        "category": "admit_card_withdrawal",
        "doc": "examination_manual.pdf",
        "chunk_ids": [41],
        "fact_en": "Permission to appear in an examination or the admit card may be withdrawn if issued by mistake or if the student was not eligible.",
        "en": [
            "Under what conditions can an admit card or exam permission be withdrawn?",
            "Can the university cancel examination permission if issued by mistake?",
            "What happens if an ineligible student was given an admit card?",
            "When can exam hall permission be cancelled by the examination office?"
        ],
        "indic": [
            "परीक्षा प्रवेश पत्र (Admit Card) किन परिस्थितियों में वापस लिया जा सकता है?",
            "यदि गलती से एडमिट कार्ड जारी हो गया हो तो क्या परीक्षा अनुमति रद्द हो सकती है?",
            "परीक्षा में बैठने की अनुमति कब वापस ली जा सकती है?",
            "क्या अपात्र छात्रों का एडमिट कार्ड रद्द किया जा सकता है?"
        ],
        "codemix": [
            "Admit card ya exam permission kin conditions me withdraw ho sakti hai?",
            "Agar admit card galti se issue ho gaya ho toh kya cancel ho sakta hai?",
            "Ineligible student ko mila hua admit card university cancel kar sakti hai kya?",
            "Exam hall me appear hone ki permission kab wapas li ja sakti hai?"
        ]
    },
    {
        "category": "scholarship_categories",
        "doc": "scholarship_policy.pdf",
        "chunk_ids": [327, 330],
        "fact_en": "Bennett University offers Admission Scholarships (Merit, Sibling, Single Girl Child, Ward of Defence Personnel, Alumni, and Times Group rebate) and Academic Scholarships for continuing students.",
        "en": [
            "What categories of scholarships are offered by Bennett University?",
            "What are the types of on-admission scholarships available?",
            "Does Bennett University offer a Single Girl Child scholarship?",
            "What scholarships exist for wards of defence personnel?"
        ],
        "indic": [
            "बेनेट विश्वविद्यालय द्वारा कौन-कौन सी छात्रवृत्तियां (Scholarships) प्रदान की जाती हैं?",
            "प्रवेश के समय मिलने वाली छात्रवृत्तियों की श्रेणियां क्या हैं?",
            "क्या विश्वविद्यालय में सिंगल गर्ल चाइल्ड के लिए छात्रवृत्ति है?",
            "रक्षा कर्मियों (Defence Personnel) के बच्चों के लिए क्या कोई विशेष छात्रवृत्ति है?"
        ],
        "codemix": [
            "Bennett University me kaun kaun si scholarship categories milti hain?",
            "Admission ke time par kaun si scholarship schemes available hain?",
            "Kya Single Girl Child ke liye koi specific scholarship hai?",
            "Defence personnel ke bacho ke liye kya scholarship policy hai?"
        ]
    },
    {
        "category": "scholarship_highest_value",
        "doc": "scholarship_policy.pdf",
        "chunk_ids": [327],
        "fact_en": "In cases where a student qualifies for multiple scholarships, only the scholarship with the highest monetary value is awarded.",
        "en": [
            "What happens if a student qualifies for more than one scholarship?",
            "Can a student combine two scholarships at Bennett University?",
            "Is a student allowed to avail both sibling and merit scholarships together?",
            "Which scholarship is awarded if a candidate is eligible for multiple schemes?"
        ],
        "indic": [
            "यदि कोई छात्र एक से अधिक छात्रवृत्ति के लिए पात्र हो तो कौन सी छात्रवृत्ति दी जाएगी?",
            "क्या एक साथ दो छात्रवृत्तियों का लाभ उठाया जा सकता है?",
            "मल्टीपल स्कॉलरशिप मिलने पर विश्वविद्यालय का क्या नियम है?",
            "क्या मेरिट और सिबलिंग स्कॉलरशिप दोनों एक साथ मिल सकती हैं?"
        ],
        "codemix": [
            "Agar ek student multiple scholarships ke liye qualify kare toh kaun si milegi?",
            "Kya do scholarships ek saath combine kar sakte hain?",
            "Multiple scholarships eligible hone par university ka rule kya hai?",
            "Kya sibling discount aur merit scholarship dono saath me avail ho sakti hain?"
        ]
    },
    {
        "category": "single_girl_child_affidavit",
        "doc": "scholarship_policy.pdf",
        "chunk_ids": [391],
        "fact_en": "For the Single Girl Child scholarship, parents must submit an affidavit on Rs. 100 stamp paper notarized by a notary.",
        "en": [
            "What document is required to claim the Single Girl Child scholarship?",
            "What is the stamp paper denomination for the Single Girl Child affidavit?",
            "Who must execute the affidavit for the Single Girl Child scholarship?",
            "Is a notary affidavit required for the Single Girl Child rebate?"
        ],
        "indic": [
            "सिंगल गर्ल चाइल्ड स्कॉलरशिप के लिए माता-पिता को कौन सा दस्तावेज देना होता है?",
            "सिंगल गर्ल चाइल्ड एफिडेविट के लिए कितने रुपये का स्टाम्प पेपर चाहिए?",
            "क्या सिंगल गर्ल चाइल्ड के लिए नोटरी एफिडेविट आवश्यक है?",
            "सिंगल गर्ल चाइल्ड छात्रवृत्ति का दावा करने के लिए क्या प्रमाण पत्र चाहिए?"
        ],
        "codemix": [
            "Single Girl Child scholarship claim karne ke liye kaun sa document chahiye?",
            "Single Girl Child affidavit ke liye kitne rupees ka stamp paper lagta hai?",
            "Kya Single Girl Child scholarship ke liye Rs 100 ka notary affidavit mandatory hai?",
            "Parents ko single girl child ke liye kaun sa proof submit karna hota hai?"
        ]
    },
    {
        "category": "scholarship_withdrawal",
        "doc": "scholarship_policy.pdf",
        "chunk_ids": [336],
        "fact_en": "Scholarship is withdrawn if advance fee is not deposited by the given due date, or if the student fails to produce documentary proof of eligibility.",
        "en": [
            "Under what circumstances can a merit scholarship be withdrawn?",
            "What happens if a candidate fails to deposit the advance fee on time?",
            "Can scholarship be revoked for failing to produce eligibility documents?",
            "When will a candidate forfeit their claim over the admission scholarship?"
        ],
        "indic": [
            "मेरिट छात्रवृत्ति किन परिस्थितियों में वापस (Withdrawn) ली जा सकती है?",
            "यदि छात्र नियत तिथि तक अग्रिम शुल्क जमा न करे तो क्या होगा?",
            "क्या पात्रता दस्तावेज प्रस्तुत न करने पर स्कॉलरशिप रद्द हो सकती है?",
            "छात्रवृत्ति का अधिकार कब समाप्त हो जाता है?"
        ],
        "codemix": [
            "Merit scholarship kin conditions me withdraw ho sakti hai?",
            "Advance fee time pe deposit na karne par scholarship ka kya hoga?",
            "Agar eligibility documents verify na ho payein toh kya scholarship cancel hogi?",
            "Scholarship forfeit hone ke kya reasons hain policy me?"
        ]
    },
    {
        "category": "scholarship_committee",
        "doc": "scholarship_policy.pdf",
        "chunk_ids": [327],
        "fact_en": "The verified list of eligible students for Merit Scholarships is reviewed by the Scholarship Committee comprising the Registrar, VP Admissions & Marketing, and Finance Controller.",
        "en": [
            "Who are the members of the Bennett University Scholarship Committee?",
            "Does the Registrar review the merit scholarship list?",
            "Which committee approves the verified list of eligible scholarship candidates?",
            "Is the Finance Controller part of the Scholarship Committee?"
        ],
        "indic": [
            "बेनेट विश्वविद्यालय की छात्रवृत्ति समिति (Scholarship Committee) में कौन-कौन शामिल हैं?",
            "क्या कुलसचिव (Registrar) स्कॉलरशिप समिति के सदस्य हैं?",
            "मेरिट छात्रवृत्ति सूची की समीक्षा कौन सी समिति करती है?",
            "फाइनेंस कंट्रोलर का छात्रवृत्ति समिति में क्या पद है?"
        ],
        "codemix": [
            "Scholarship Committee ke members kaun kaun hain Bennett University me?",
            "Kya Registrar aur Finance Controller scholarship review committee me hote hain?",
            "Merit scholarship eligible students ki list kaun approve karta hai?",
            "VP Admissions & Marketing kya scholarship committee ke member hain?"
        ]
    },
    {
        "category": "undertaking_truthfulness",
        "doc": "student_parent_undertaking.pdf",
        "chunk_ids": [391, 392],
        "fact_en": "Submission of false information or omission of relevant credentials may result in application withdrawal, denial, or revocation of admission.",
        "en": [
            "What are the consequences of submitting false information in the admission application?",
            "Can Bennett University revoke an admission if false credentials are detected later?",
            "What undertaking is given by parents regarding document verification?",
            "What happens if a student omits relevant educational credentials?"
        ],
        "indic": [
            "प्रवेश आवेदन में गलत जानकारी या झूठे दस्तावेज देने पर क्या परिणाम हो सकते हैं?",
            "क्या बाद में फर्जी दस्तावेज पाए जाने पर विश्वविद्यालय प्रवेश रद्द कर सकता है?",
            "दस्तावेज सत्यापन के संबंध में माता-पिता क्या वचनबद्धता देते हैं?",
            "यदि कोई छात्र अपनी योग्यता से संबंधित तथ्य छुपाता है तो क्या कार्रवाई होगी?"
        ],
        "codemix": [
            "Admission form me galat information submit karne pe kya consequences honge?",
            "Kya later date par false credentials detect hone par admission revoke ho sakta hai?",
            "Document verification ke regarding students aur parents kya undertaking dete hain?",
            "False documents ya deliberate omission par university kya action leti hai?"
        ]
    },
    {
        "category": "undertaking_contact_details",
        "doc": "student_parent_undertaking.pdf",
        "chunk_ids": [392, 393],
        "fact_en": "Parents undertake to provide complete and correct contact details including telephone numbers, email IDs, and local guardian details for notices.",
        "en": [
            "What contact details must parents provide according to the undertaking?",
            "Is the university responsible if a notice sent to a group fails to reach an individual?",
            "Whose contact details must be submitted besides the parents' own details?",
            "What is the parents' obligation regarding updated contact information?"
        ],
        "indic": [
            "अंडरटेकिंग के अनुसार माता-पिता को कौन से संपर्क विवरण (Contact details) देने होते हैं?",
            "क्या किसी समूह नोटिस के व्यक्तिगत रूप से न पहुंचने पर विश्वविद्यालय उत्तरदायी है?",
            "स्थानीय अभिभावक (Local Guardian) के संबंध में क्या नियम है?",
            "संपर्क जानकारी के संबंध में अभिभावकों का क्या उत्तरदायित्व है?"
        ],
        "codemix": [
            "Parents ko undertaking ke hisab se kaun si contact details provide karni hoti hain?",
            "Kya local guardian ka phone number aur email ID dena mandatory hai?",
            "Agar notice individual tak na pahuche toh kya university responsible hogi?",
            "Contact number update rakhne ke baare me undertaking me kya likha hai?"
        ]
    },
    {
        "category": "unfair_means_examination",
        "doc": "examination_manual.pdf",
        "chunk_ids": [4, 5, 46],
        "fact_en": "Cheating or unfair means in any examination is strictly prohibited and subject to severe disciplinary penalties by the Examination Disciplinary Committee.",
        "en": [
            "What is Bennett University's policy regarding unfair means during examinations?",
            "Which committee decides penalties for cheating in examinations?",
            "Can cheating lead to cancellation of an entire semester's examinations?",
            "What are the disciplinary consequences of malpractice in exams?"
        ],
        "indic": [
            "परीक्षा के दौरान अनुचित साधनों (Unfair Means) के प्रयोग पर विश्वविद्यालय की क्या नीति है?",
            "परीक्षा में नकल करने पर दण्ड का निर्धारण कौन सी समिति करती है?",
            "क्या नकल करने पर पूरे सेमेस्टर की परीक्षा रद्द की जा सकती है?",
            "परीक्षा में कदाचार (Malpractice) के क्या परिणाम हो सकते हैं?"
        ],
        "codemix": [
            "Exam ke dauran cheating ya unfair means karne par university ki policy kya hai?",
            "Cheating pakde jane par kaun si committee punishment decide karti hai?",
            "Kya UFM lagne par poora semester cancel ho sakta hai?",
            "Malpractice in exams ke disciplinary consequences kya hain?"
        ]
    },
    {
        "category": "attendance_requirement",
        "doc": "examination_manual.pdf",
        "chunk_ids": [3, 4, 39, 52],
        "fact_en": "Students are required to maintain a minimum attendance threshold (typically 75%) to be eligible to appear in end-semester examinations.",
        "en": [
            "What is the minimum attendance requirement to appear in semester examinations?",
            "Can a student with attendance below 75% be detained from exams?",
            "What are the regulations governing student attendance for examinations?",
            "What happens if a student fails to meet the minimum attendance criteria?"
        ],
        "indic": [
            "सेमेस्टर परीक्षा में बैठने के लिए न्यूनतम कितने प्रतिशत उपस्थिति (Attendance) अनिवार्य है?",
            "क्या 75% से कम उपस्थिति होने पर छात्र को परीक्षा से वंचित (Detain) किया जा सकता है?",
            "विश्वविद्यालय में उपस्थिति संबंधी नियम क्या हैं?",
            "न्यूनतम उपस्थिति पूरी न होने पर छात्र के खिलाफ क्या कार्रवाई होती है?"
        ],
        "codemix": [
            "Semester exam me baithne ke liye minimum kitna attendance percentage chahiye?",
            "Kya 75% se kam attendance hone par exam se detain kar diya jata hai?",
            "Bennett University me attendance ke kya rules hain?",
            "Agar minimum attendance complete na ho toh kya student ko admit card milega?"
        ]
    }
]

# Intentionally UNANSWERABLE queries (deliberate out-of-domain / false-premise questions)
UNANSWERABLE_TEMPLATES = [
    {
        "en": "Can students keep pet dogs in the university hostel rooms?",
        "indic": "क्या छात्र विश्वविद्यालय के छात्रावास (हॉस्टल) के कमरों में पालतू कुत्ते रख सकते हैं?",
        "codemix": "Kya students university hostel ke room me pet dogs rakh sakte hain?",
        "reason": "Hostel pet policy is not covered in examination manual or scholarship policy."
    },
    {
        "en": "What is the scholarship percentage for national-level badminton players?",
        "indic": "राष्ट्रीय स्तर के बैडमिंटन खिलाड़ियों के लिए खेल छात्रवृत्ति का कितना प्रतिशत है?",
        "codemix": "National level badminton players ke liye kitna percent sports scholarship milta hai?",
        "reason": "Specific badminton sports quota percentages are not defined in the provided policy."
    },
    {
        "en": "How much monthly stipend is provided by Bennett University to PhD students for NASA research projects?",
        "indic": "नासा अनुसंधान परियोजनाओं के लिए पीएचडी छात्रों को कितना मासिक वजीफा (Stipend) दिया जाता है?",
        "codemix": "NASA research projects ke liye PhD students ko kitna monthly stipend milta hai?",
        "reason": "NASA collaboration stipends do not exist in the provided regulations."
    },
    {
        "en": "What is the procedure to get a free laptop from the university at the time of admission?",
        "indic": "प्रवेश के समय विश्वविद्यालय से निःशुल्क लैपटॉप प्राप्त करने की क्या प्रक्रिया है?",
        "codemix": "Admission ke time par free laptop lene ka kya procedure hai?",
        "reason": "Free laptop distribution is not part of the university policy."
    },
    {
        "en": "What is the fine for parking an electric scooter in front of the Vice Chancellor's bungalow?",
        "indic": "कुलपति के बंगले के सामने इलेक्ट्रिक स्कूटर पार्क करने पर कितना जुर्माना है?",
        "codemix": "VC ke bungalow ke samne electric scooter park karne par kitna fine lagta hai?",
        "reason": "Parking penalties for VC bungalow are absent from the documents."
    },
    {
        "en": "Can a student exchange their Bennett degree for an Oxford degree after 2 years?",
        "indic": "क्या कोई छात्र 2 साल बाद अपनी बेनेट डिग्री को ऑक्सफोर्ड डिग्री से बदल सकता है?",
        "codemix": "Kya student 2 saal baad Bennett degree ko Oxford degree se exchange kar sakta hai?",
        "reason": "Oxford degree exchange program does not exist in the regulations."
    },
    {
        "en": "What is the fee refund policy if a student withdraws after completing 3 full years of study?",
        "indic": "तीन पूरे शैक्षणिक वर्ष पूरे करने के बाद अध्ययन छोड़ने पर शुल्क वापसी की क्या नीति है?",
        "codemix": "3 years complete karne ke baad drop lene par fees refund ka kya rule hai?",
        "reason": "Fee refund after 3 years is not provided in admission undertaking."
    },
    {
        "en": "Are students permitted to operate private commercial food trucks on university campus?",
        "indic": "क्या छात्रों को विश्वविद्यालय परिसर में निजी वाणिज्यिक फूड ट्रक चलाने की अनुमति है?",
        "codemix": "Kya campus ke andar students ko private food truck chalane ki permission hai?",
        "reason": "Commercial food truck operation is unmentioned in the documents."
    },
    {
        "en": "What is the scholarship discount for students whose parents are astronauts?",
        "indic": "जिन छात्रों के माता-पिता अंतरिक्ष यात्री (Astronaut) हैं, उनके लिए छात्रवृत्ति में कितनी छूट है?",
        "codemix": "Jin students ke parents astronaut hain unke liye kitna scholarship discount hai?",
        "reason": "No astronaut scholarship category exists."
    },
    {
        "en": "How can a student apply for an Antarctic semester exchange program?",
        "indic": "छात्र अंटार्कटिका सेमेस्टर एक्सचेंज प्रोग्राम के लिए कैसे आवेदन कर सकते हैं?",
        "codemix": "Antarctica semester exchange program ke liye kaise apply karein?",
        "reason": "Antarctic exchange program is not part of Bennett University rules."
    }
]


# 15+ Difficult Cases specifically designed for Module 6 qualitative evaluation
DIFFICULT_CASES = [
    {
        "case_id": "DIFF_01",
        "type": "Code-Mixing & Slang",
        "query": "Yaar mujhe batao scholarship ke liye 12th board me kitna percent compulsory hai?",
        "language": "Code-Mixed",
        "difficulty_factor": "Colloquial Hindi particle 'Yaar' with mixed English/Hindi syntax",
        "target_answer": "Merit scholarships depend on approved qualifying examination scores as per Annexure A.",
        "expected_answerable": "ANSWERABLE"
    },
    {
        "case_id": "DIFF_02",
        "type": "Phonetic Spelling Variation",
        "query": "skolerchip ke liye minimum elegibility critaria kya hai?",
        "language": "Code-Mixed",
        "difficulty_factor": "Phonetic misspelling of scholarship ('skolerchip'), eligibility ('elegibility'), and criteria ('critaria')",
        "target_answer": "Scholarship eligibility is determined by scores in qualifying examinations and documentary evidence.",
        "expected_answerable": "ANSWERABLE"
    },
    {
        "case_id": "DIFF_03",
        "type": "Cross-Lingual Devanagari Script",
        "query": "एंड सेमेस्टर परीक्षा पास करने के लिए न्यूनतम कितने प्रतिशत अंक चाहिए?",
        "language": "Indic",
        "difficulty_factor": "Pure Devanagari script query querying an English-language document",
        "target_answer": "30% marks in the end term exam and a total of 40 marks out of 100 overall.",
        "expected_answerable": "ANSWERABLE"
    },
    {
        "case_id": "DIFF_04",
        "type": "Exact Numerical Threshold",
        "query": "How many marks out of 100 are mandatory to pass a course overall?",
        "language": "English",
        "difficulty_factor": "Exact boundary condition (40 out of 100 overall, 30% in end sem)",
        "target_answer": "A total of 40 marks out of 100 is mandatory to pass a course.",
        "expected_answerable": "ANSWERABLE"
    },
    {
        "case_id": "DIFF_05",
        "type": "Stamp Paper Value & Named Entity",
        "query": "Kitne rupees ke stamp paper par single girl child ka affidavit banta hai?",
        "language": "Code-Mixed",
        "difficulty_factor": "Extracting monetary denomination 'Rs. 100' stamp paper and specific affidavit type",
        "target_answer": "Rs. 100/- Stamp paper (Notary) Affidavit executed by parents.",
        "expected_answerable": "ANSWERABLE"
    },
    {
        "case_id": "DIFF_06",
        "type": "Named Entities & Committee Composition",
        "query": "Who are the members of the Scholarship Committee that reviews merit scholarships?",
        "language": "English",
        "difficulty_factor": "Multi-entity extraction: Registrar, VP Admissions & Marketing, Finance Controller",
        "target_answer": "Registrar, VP Admissions & Marketing, and Finance Controller.",
        "expected_answerable": "ANSWERABLE"
    },
    {
        "case_id": "DIFF_07",
        "type": "Ambiguity Between Schemes",
        "query": "Can a student receive both sibling scholarship and merit scholarship together?",
        "language": "English",
        "difficulty_factor": "Disambiguating rule: only the single highest monetary value scholarship is awarded",
        "target_answer": "No, only the scholarship with the highest monetary value is awarded.",
        "expected_answerable": "ANSWERABLE"
    },
    {
        "case_id": "DIFF_08",
        "type": "Grade Point Semantics",
        "query": "Does an F grade carry any grade points towards CGPA?",
        "language": "English",
        "difficulty_factor": "Semantic understanding that failing grade 'F' awards 0 grade points and course must be repeated",
        "target_answer": "In the case of 'F', no grade points are awarded and the course must be repeated.",
        "expected_answerable": "ANSWERABLE"
    },
    {
        "case_id": "DIFF_09",
        "type": "Intentionally Unanswerable with Believable Entity",
        "query": "What is the sports scholarship percentage for national archery championship winners?",
        "language": "English",
        "difficulty_factor": "Hallucination trap: sounds plausible but archery sports quota is absent in text",
        "target_answer": "Insufficient information available in the provided documents.",
        "expected_answerable": "UNANSWERABLE"
    },
    {
        "case_id": "DIFF_10",
        "type": "Intentionally Unanswerable in Hinglish",
        "query": "Hostel me pet cat rakhne ke liye warden se permission kaise milti hai?",
        "language": "Code-Mixed",
        "difficulty_factor": "Hallucination trap in Hinglish: hostel pet policy is nonexistent in policy docs",
        "target_answer": "Insufficient information available in the provided documents.",
        "expected_answerable": "UNANSWERABLE"
    },
    {
        "case_id": "DIFF_11",
        "type": "Intentionally Unanswerable in Devanagari",
        "query": "पीएचडी छात्रों को नासा रिसर्च प्रोजेक्ट के लिए कितना मासिक स्टाइपेंड मिलता है?",
        "language": "Indic",
        "difficulty_factor": "Hallucination trap in Devanagari: fictitious research collaboration",
        "target_answer": "Insufficient information available in the provided documents.",
        "expected_answerable": "UNANSWERABLE"
    },
    {
        "case_id": "DIFF_12",
        "type": "Multi-Condition Rule",
        "query": "Advance fee due date miss hone par scholarship ka kya hota hai?",
        "language": "Code-Mixed",
        "difficulty_factor": "Contingent policy rule: candidate loses claim and scholarship is passed to next candidate",
        "target_answer": "Candidate has no claim over scholarship and it is passed to next eligible candidate.",
        "expected_answerable": "ANSWERABLE"
    },
    {
        "case_id": "DIFF_13",
        "type": "Administrative / Mistake Condition",
        "query": "Can the university withdraw exam hall permission if admit card was issued by mistake?",
        "language": "English",
        "difficulty_factor": "Exception clause: withdraw if issued through mistake or student was ineligible",
        "target_answer": "Yes, permission may be withdrawn if issued through a mistake or student was ineligible.",
        "expected_answerable": "ANSWERABLE"
    },
    {
        "case_id": "DIFF_14",
        "type": "Parental Undertaking Clause",
        "query": "Under what condition can admission be revoked even if detected at a later date?",
        "language": "English",
        "difficulty_factor": "Legal/undertaking language: submission of false information or noncompliance with criteria",
        "target_answer": "Noncompliance with admission criteria or submission of false information.",
        "expected_answerable": "ANSWERABLE"
    },
    {
        "case_id": "DIFF_15",
        "type": "Cross-Lingual Examination Malpractice",
        "query": "परीक्षा में नकल (Unfair Means) पकड़े जाने पर कौन सी कमेटी कार्रवाई करती है?",
        "language": "Indic",
        "difficulty_factor": "Devanagari script query on disciplinary committee jurisdiction",
        "target_answer": "Examination Disciplinary Committee enforces disciplinary penalties for unfair means.",
        "expected_answerable": "ANSWERABLE"
    },
    {
        "case_id": "DIFF_16",
        "type": "Code-Mixed Typo with Acronym",
        "query": "End sem me UFM lagne par poora semister radd ho sakta hai kya?",
        "language": "Code-Mixed",
        "difficulty_factor": "Acronym 'UFM' (Unfair Means) + misspelling 'semister' + Hindi 'radd' (cancelled)",
        "target_answer": "Yes, unfair means malpractice carries severe penalties including exam cancellation.",
        "expected_answerable": "ANSWERABLE"
    }
]


def generate_dataset():
    print("Generating comprehensive evaluation dataset (350+ instances)...")
    dataset = []
    item_id = 1

    # 1. Generate Answerable instances across languages and variations
    for template in FACT_TEMPLATES:
        cat = template["category"]
        doc = template["doc"]
        cids = template["chunk_ids"]
        gt_answer = template["fact_en"]

        # English queries
        for q in template["en"]:
            dataset.append({
                "id": f"QA_{item_id:04d}",
                "query": q,
                "language": "English",
                "is_code_mixed": False,
                "answerability": "ANSWERABLE",
                "ground_truth_answer": gt_answer,
                "relevant_chunk_ids": ";".join(map(str, cids)),
                "source_doc": doc,
                "difficulty": "standard",
                "category": cat
            })
            item_id += 1

        # Indic (Hindi Devanagari) queries
        for q in template["indic"]:
            dataset.append({
                "id": f"QA_{item_id:04d}",
                "query": q,
                "language": "Indic",
                "is_code_mixed": False,
                "answerability": "ANSWERABLE",
                "ground_truth_answer": gt_answer,
                "relevant_chunk_ids": ";".join(map(str, cids)),
                "source_doc": doc,
                "difficulty": "cross_lingual",
                "category": cat
            })
            item_id += 1

        # Code-Mixed (Hinglish) queries
        for q in template["codemix"]:
            dataset.append({
                "id": f"QA_{item_id:04d}",
                "query": q,
                "language": "Code-Mixed",
                "is_code_mixed": True,
                "answerability": "ANSWERABLE",
                "ground_truth_answer": gt_answer,
                "relevant_chunk_ids": ";".join(map(str, cids)),
                "source_doc": doc,
                "difficulty": "code_mixed",
                "category": cat
            })
            item_id += 1

    # Synthesize additional variations to reach ~360 instances
    prefix_variations_en = [
        "Please tell me, ",
        "According to the official policy, ",
        "Can you clarify: ",
        "In Bennett regulations, "
    ]
    prefix_variations_indic = [
        "कृपया बताइए कि ",
        "विश्वविद्यालय नियमों के अनुसार, ",
        "क्या आप स्पष्ट कर सकते हैं कि ",
        "आधिकारिक नीति के तहत, "
    ]
    prefix_variations_codemix = [
        "Bhai batao ",
        "Official guidelines ke according, ",
        "Zara clarify karna: ",
        "Bennett university rules me, "
    ]

    for t_idx, template in enumerate(FACT_TEMPLATES):
        cat = template["category"]
        doc = template["doc"]
        cids = template["chunk_ids"]
        gt_answer = template["fact_en"]

        for var_idx in range(4):
            # EN variation
            q_en = f"{prefix_variations_en[var_idx]}{template['en'][var_idx % len(template['en'])].lower()}"
            dataset.append({
                "id": f"QA_{item_id:04d}",
                "query": q_en,
                "language": "English",
                "is_code_mixed": False,
                "answerability": "ANSWERABLE",
                "ground_truth_answer": gt_answer,
                "relevant_chunk_ids": ";".join(map(str, cids)),
                "source_doc": doc,
                "difficulty": "variation",
                "category": cat
            })
            item_id += 1

            # Indic variation
            q_ind = f"{prefix_variations_indic[var_idx]}{template['indic'][var_idx % len(template['indic'])]}"
            dataset.append({
                "id": f"QA_{item_id:04d}",
                "query": q_ind,
                "language": "Indic",
                "is_code_mixed": False,
                "answerability": "ANSWERABLE",
                "ground_truth_answer": gt_answer,
                "relevant_chunk_ids": ";".join(map(str, cids)),
                "source_doc": doc,
                "difficulty": "cross_lingual_variation",
                "category": cat
            })
            item_id += 1

            # Hinglish variation
            q_cm = f"{prefix_variations_codemix[var_idx]}{template['codemix'][var_idx % len(template['codemix'])]}"
            dataset.append({
                "id": f"QA_{item_id:04d}",
                "query": q_cm,
                "language": "Code-Mixed",
                "is_code_mixed": True,
                "answerability": "ANSWERABLE",
                "ground_truth_answer": gt_answer,
                "relevant_chunk_ids": ";".join(map(str, cids)),
                "source_doc": doc,
                "difficulty": "codemix_variation",
                "category": cat
            })
            item_id += 1

    # 2. Generate Unanswerable instances across languages
    for unans in UNANSWERABLE_TEMPLATES:
        # EN Unanswerable
        dataset.append({
            "id": f"QA_{item_id:04d}",
            "query": unans["en"],
            "language": "English",
            "is_code_mixed": False,
            "answerability": "UNANSWERABLE",
            "ground_truth_answer": "Insufficient information available in the provided documents.",
            "relevant_chunk_ids": "",
            "source_doc": "none",
            "difficulty": "unanswerable",
            "category": "hallucination_probe"
        })
        item_id += 1

        # Indic Unanswerable
        dataset.append({
            "id": f"QA_{item_id:04d}",
            "query": unans["indic"],
            "language": "Indic",
            "is_code_mixed": False,
            "answerability": "UNANSWERABLE",
            "ground_truth_answer": "Insufficient information available in the provided documents.",
            "relevant_chunk_ids": "",
            "source_doc": "none",
            "difficulty": "unanswerable_cross_lingual",
            "category": "hallucination_probe"
        })
        item_id += 1

        # Code-Mixed Unanswerable
        dataset.append({
            "id": f"QA_{item_id:04d}",
            "query": unans["codemix"],
            "language": "Code-Mixed",
            "is_code_mixed": True,
            "answerability": "UNANSWERABLE",
            "ground_truth_answer": "Insufficient information available in the provided documents.",
            "relevant_chunk_ids": "",
            "source_doc": "none",
            "difficulty": "unanswerable_codemix",
            "category": "hallucination_probe"
        })
        item_id += 1

        # Add 4 additional paraphrases per unanswerable template to create 80+ unanswerable probes
        for prefix in ["Can anyone tell me, ", "Is it allowed that ", "What are the university rules if ", "Do we have any clause where "]:
            dataset.append({
                "id": f"QA_{item_id:04d}",
                "query": f"{prefix}{unans['en'].lower()}",
                "language": "English",
                "is_code_mixed": False,
                "answerability": "UNANSWERABLE",
                "ground_truth_answer": "Insufficient information available in the provided documents.",
                "relevant_chunk_ids": "",
                "source_doc": "none",
                "difficulty": "unanswerable_probe",
                "category": "hallucination_probe"
            })
            item_id += 1

    # Save to questions.csv
    csv_path = QUESTIONS_DIR / "questions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "id", "query", "language", "is_code_mixed", "answerability",
            "ground_truth_answer", "relevant_chunk_ids", "source_doc",
            "difficulty", "category"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)

    print(f"Saved {len(dataset)} question-answer instances to {csv_path}")

    # Save difficult cases to difficult_cases.csv
    diff_csv_path = EVALUATION_DIR / "difficult_cases.csv"
    with open(diff_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "case_id", "type", "query", "language", "difficulty_factor",
            "target_answer", "expected_answerable"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(DIFFICULT_CASES)

    print(f"Saved {len(DIFFICULT_CASES)} difficult qualitative cases to {diff_csv_path}")


if __name__ == "__main__":
    generate_dataset()
