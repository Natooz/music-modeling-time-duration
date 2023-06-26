"""
Constants file
"""
from torch.cuda import is_available as cuda_available


SEED = 444

# Tokenizer params (same as MidiTok expect for new constants)
PITCH_RANGE = range(21, 109)
BEAT_RES = {(0, 1): 8, (1, 2): 4, (2, 4): 2, (4, 8): 1}
NB_VELOCITIES = 8
ADDITIONAL_TOKENS = {'Chord': False, 'Rest': False, 'Tempo': False, 'Program': False, 'TimeSignature': False,
                     'rest_range': (2, 8), 'nb_tempos': 32, 'tempo_range': (40, 250), 'time_signature_range': (8, 2)}
SPECIAL_TOKENS = ["PAD", "MASK", "BOS", "EOS", "SEP"]
TOKENIZER_PARAMS = {'pitch_range': PITCH_RANGE, 'beat_res': BEAT_RES, 'nb_velocities': NB_VELOCITIES,
                    'additional_tokens': ADDITIONAL_TOKENS, "special_tokens": SPECIAL_TOKENS}
TIME_DIVISION = 384
DATA_AUGMENTATION_OFFSETS = (2, 1, 0)
BPE_NB_FILES_LIM = 5000
VOCAB_SIZE_BPE_GEN = 2000
VOCAB_SIZE_BPE_CLA = 5000  # also for CON
VOCAB_SIZE_BPE_TCP = 0

# For classification
NOTE_DENSITY_RANGE = range(0, 13)  # in notes per beat

# For transcription
SAMPLE_RATE = 16000
N_FFT = 2048
WIN_LENGTH = 2048
HOP_WIDTH = 128
N_MELS = 512
SAFE_LOG_EPS = 1e-4
ONSET_TOLERANCE = 1
TEMPO = 120

# Transformer config (for classification and contrastive)
MODEL_DIM = 768
MODEL_NB_HEADS = 12
MODEL_D_FFWD = MODEL_DIM * 4
MODEL_NB_LAYERS = 12
MODEL_NB_POS_ENC_PARAMS = 2048  # params for positional encoding positions

# Transformer config (for transcription)
TCP_DIM = 512
TCP_D_KV = 64
TCP_NB_HEADS_ENCODER = 8
TCP_NB_HEADS_DECODER = 8
TCP_D_FFWD = TCP_DIM * 4
TCP_NB_LAYERS_ENCODER = 8
TCP_NB_LAYERS_DECODER = 8
TCP_NB_POS_ENC_PARAMS = 2048


# COMMON TRAINING PARAMS
DROPOUT = 0.1
EVAL_ACCUMULATION_STEPS = None  # to use in case of CUDA OOM during eval
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 3.0
LABEL_SMOOTHING = 0.0
VALID_SPLIT = 0.10
TEST_SPLIT = 0.15
USE_CUDA = True
USE_AMP = True
USE_GRADIENT_CHECKPOINTING = True
DDP_FIND_UNUSED_PARAMETERS = False
DDP_BUCKET_CAP_MB = None  # default to 25mb
VALID_INTVL = 1000
LOG_STEPS_INTVL = 20
SAVE_STEPS = 1000
SAVE_TOTAL_LIMIT = 1
WARMUP_RATIO = 0.3

# TRAINING PARAMS GEN
BATCH_SIZE_GEN = 24
GRAD_ACC_STEPS_GEN = 2
MIN_SEQ_LEN_GEN = 384
MAX_SEQ_LEN_GEN = 512
TRAINING_STEPS_GEN = 50000
LEARNING_RATE_GEN = 1e-4
LR_SCHEDULER_GEN = 'cosine_with_restarts'

# TRAINING PARAMS PRETRAINING
BATCH_SIZE_PT = 48
GRAD_ACC_STEPS_PT = 1
MIN_SEQ_LEN_PT = 384
MAX_SEQ_LEN_PT = 512
TRAINING_STEPS_PT = 100000
LEARNING_RATE_PT = 1e-4
LR_SCHEDULER_PT = 'cosine_with_restarts'
MASK_RATIO_CLA_PT = 0.15

# TRAINING PARAMS CLA FT
BATCH_SIZE_CLA_FT = 48
GRAD_ACC_STEPS_CLA = 1
MIN_SEQ_LEN_CLA_FT = 384
MAX_SEQ_LEN_CLA_FT = 512
TRAINING_STEPS_CLA_FT = 50000
LEARNING_RATE_CLA_FT = 3e-5
LR_SCHEDULER_CLA = 'cosine_with_restarts'

# TRAINING PARAMS CONTRASTIVE
BATCH_SIZE_CON = 48
GRAD_ACC_STEPS_CON = 1
MIN_SEQ_LEN_CON = 384
MAX_SEQ_LEN_CON = 512
LEARNING_RATE_CON = 3e-5
TRAINING_STEPS_CON = 50000
LR_SCHEDULER_CON = 'cosine_with_restarts'
POOLER_TYPE_CON = "cls"
TEMPERATURE_CON = 0.05

# TRAINING PARAMS TRANSCRIPTION
BATCH_SIZE_TCP = 128
GRAD_ACC_STEPS_TCP = 1
SEQ_LEN_ENCODER_TCP = 512
MIN_SEQ_LEN_DECODER_TCP = 32
MAX_SEQ_LEN_DECODER_TCP = 256
LEARNING_RATE_TCP = 3e-4
TRAINING_STEPS_TCP = 50000
LR_SCHEDULER_TCP = 'cosine_with_restarts'
USE_AMP_TCP = False

# TEST PARAMS GEN
NB_INFERENCES_GEN = 512
MIN_SEQ_LEN_TEST_GEN = 256
MAX_SEQ_LEN_TEST_GEN = 512
BATCH_SIZE_TEST_GEN = 92
NUM_BEAMS = 1  # in practice the generation will use a batch size = BATCH_SIZE_TEST * NUM_BEAMS
TEMPERATURE_SAMPLING = 1.
TOP_K = 15
TOP_P = 0.95
EPSILON_CUTOFF = None
ETA_CUTOFF = None

# TEST PARAMS CLA
MIN_SEQ_LEN_TEST_CLA = 384
MAX_SEQ_LEN_TEST_CLA = 512
BATCH_SIZE_TEST_CLA = 32

# TEST PARAMS CON
MIN_SEQ_LEN_TEST_CON = 384
MAX_SEQ_LEN_TEST_CON = 512
BATCH_SIZE_TEST_CON = 32
AUGMENTATIONS_TESTS_CON = {"pitch_+1": (1, 0, 0),
                           "pitch_+2": (2, 0, 0),
                           "pitch_+12": (12, 0, 0),
                           "pitch_+24": (24, 0, 0),
                           "pitch_-1": (-1, 0, 0),
                           "pitch_-2": (-2, 0, 0),
                           "pitch_-12": (-12, 0, 0),
                           "pitch_-24": (-24, 0, 0),
                           "velocity_+1": (0, 1, 0),
                           "velocity_+2": (0, 2, 0),
                           "velocity_-1": (0, -1, 0),
                           "velocity_-2": (0, -2, 0),
                           "pitch_+12_velocity_+1": (12, 1, 0)}

# TEST PARAMS TCP
MIN_SEQ_LEN_DECODER_TEST_TCP = 384
MAX_SEQ_LEN_DECODER_TEST_TCP = 512
BATCH_SIZE_TEST_TCP = 192

# EXCEPTION ARGUMENTS EMOTION
MIN_SEQ_LEN_CLA_EMOTION = 100
MAX_SEQ_LEN_CLA_EMOTION = 256
MIN_SEQ_LEN_TEST_CLA_EMOTION = 100
MAX_SEQ_LEN_TEST_CLA_EMOTION = 256
TRAINING_STEPS_PT_EMOTION = 40000
TRAINING_STEPS_CLA_FT_EMOTION = 15000


# in case no GPU is available
if not cuda_available():
    USE_AMP = USE_CUDA = False
