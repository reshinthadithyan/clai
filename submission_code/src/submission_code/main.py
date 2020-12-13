from transformers import pipeline

def load_transformers(model_path):
    Model = pipeline("text2text-generation", model=model_path, tokenizer=model_path)
    return Model


model_path = r"//home/reshinth-adith/reshinth/clai-helper/t5_temp/bart/best_tfmr"
Model = load_transformers(model_path)
TYPE = "TEMPLATE"

if TYPE =="TEMPLATE":
    from data_utils import nl_to_partial_tokens,cm_to_partial_tokens
    from nlp_tools import tokenizer
    from bashlint import data_tools

    def Cust_NL_Tokenizer(String,parse="Template"):
        """Custom NL Tokenizer"""
        Tokens_List = nl_to_partial_tokens(String,tokenizer=tokenizer.ner_tokenizer)
        return " ".join(Tokens_List)






def predict_transformers(invocations):
    commands = []
    for invocation in invocations:
        if not TYPE == "TEMPLATE":
            input_text = invocation
        else:
            input_text = Cust_NL_Tokenizer(invocation)
        predicted = Model(input_text)[0]["generated_text"]
        predicted_list = [predicted]#*5
        commands.append(predicted_list)
    return commands

def predict(invocations, result_cnt=5):
    """ 
    Function called by the evaluation script to interface the participants model
    `predict` function accepts the natural language invocations as input, and returns
    the predicted commands along with confidences as output. For each invocation, 
    `result_cnt` number of predicted commands are expected to be returned.
    
    Args:
        1. invocations : `list (str)` : list of `n_batch` (default 16) natural language invocations
        2. result_cnt : `int` : number of predicted commands to return for each invocation

    Returns:
        1. commands : `list [ list (str) ]` : a list of list of strings of shape (n_batch, result_cnt)
        2. confidences: `list[ list (float) ]` : confidences corresponding to the predicted commands
                                                 confidence values should be between 0.0 and 1.0. 
                                                 Shape: (n_batch, result_cnt)
    """

    n_batch = len(invocations)
    
    # `commands` and `confidences` have shape (n_batch, result_cnt)
    commands = [] #[ [''] * result_cnt for _ in range(n_batch)]
    confidences = [ 
        [1.0] * result_cnt
        for _ in range(n_batch)
    ]

    ################################################################################################
    #     Participants should add their codes to fill predict `commands` and `confidences` here    #
    ################################################################################################
    commands = predict_transformers(invocations)
    ################################################################################################
    #                               Participant code block ends                                    #
    ################################################################################################

    return commands, confidences
