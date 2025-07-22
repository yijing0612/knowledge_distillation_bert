from models.student_model import StudentClassifier
from distillation.trainer import distill_train
from sentence_transformers import SentenceTransformer
from preprocessing.proprocess_agnews import load_and_tokenize_dataset
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = load_and_tokenize_dataset(
    teacher_model_name="sentence-transformers/all-MiniLM-L6-v2",
    student_model_name="bert-base-uncased",
    max_length=128,
    save_path="data/tokenized_agnews_small", 
    small_subset=True,                         
    samples_per_class=100      ,
    mode="distill"               
)

student = StudentClassifier("bert-large-uncased")
teacher = SentenceTransformer("all-MiniLM-L6-v2")

distill_train(
    student_model=student,
    teacher_model=teacher,
    dataset=dataset,
    device=device,
    use_wandb=False
)