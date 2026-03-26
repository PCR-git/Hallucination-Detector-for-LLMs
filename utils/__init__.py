from .hooks import get_hook_q, get_hook_k, get_res_hook
from .features import get_p_tot_log, get_head_magnitudes, get_logit_feats, extract_features
from .data_utils import load_trivia_snippets, get_random_trivia_entries, generate_trivia_features, generate_sequential_training_data, merge_trivia_chunks
from .misc import set_seed
# from .rnn_data_utils import generate_trivia_sequence_features, extract_sequence_features, generate_sequential_training_data_sequences