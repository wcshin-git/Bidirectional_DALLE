from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase
from pathlib import Path

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

trainer = BpeTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"])

tokenizer.normalizer = Lowercase()

tokenizer.pre_tokenizer = Whitespace()

root = Path('datasets/MNIST64x64_Stage2/mnist_train_text')
files = [str(path) for path in root.glob('*.txt')]

tokenizer.train(files, trainer)

tokenizer.save("BPE/mnist_bpe.json")