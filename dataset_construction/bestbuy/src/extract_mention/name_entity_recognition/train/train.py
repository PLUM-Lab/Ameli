from bestbuy.src.extract_mention.name_entity_recognition.train.ner_data_maker import NERDataMaker
from bestbuy.src.extract_mention.name_entity_recognition.util.data_util import convert_cprod1_to_rasa

def gen_temp_data():
    raw_text = """
    [40"](display_size) [LED](display_type) TV
    Specifications: [16″](display_size) HD READY [LED](display_type) TV.
    [1 Year](warranty) Warranty
    Rowa [29"](display_size) [LED](display_type) TV
    Highlights:- 48"Full HD [LED](display_type) TV Triple Protection
    [80cm](display_size) (32) HD Flat TV K4000 Series 4
    [32"](display_size) LED, [2 yrs](warranty) full warranty, All care protection, Integrated Sound Station- Tweeter/20w, Family tv 2.0, Louvre Desing, Mega dynamic contract ratio, Hyper real engine, USB movie
    CG 32D0003 [LED](display_type) TV
    Screen Size : [43″](display_size)
    Resolution : 1920*1080p
    Response time : [8ms](response_time)
    USB : Yes (Music+Photo+Movie)
    Analog AV Out : Yes
    Power Supply : 110~240V 50-60Hz
    WEGA [32 Inch](display_size) SMART DLED TV HI Sound Double Glass - (Black)
    Model: [32"](display_size) Smart DLED TV HI Sound
    Hisense HX32N2176 [32"Inch](display_size) Full HD [Led](display_type) Tv
    [32 Inch](display_size) [1366x768](display_resolution) pixels HD LED TV
    [43 inch](display_size) [LED](display_type) TV
    [2 Years](warranty) Warranty & 1 Year Service Warranty
    [1920 X 1080](display_resolution) Full HD
    [Technos](brand) [39 Inch](display_size) Curved Smart [LED](display_type) TV E39DU2000 With Wallmount
    24″ Led Display Stylish Display Screen resolution : [1280 × 720](display_resolution) (HD Ready) USB : Yes VGS : Yes
    Technos 24K5 [24 Inch](display_size) LED TV
    Technos Led Tv [18.5″ Inch](display_size) (1868tw)
    [18.5 inch](display_size) stylish LED dsiplay [1280 x 720p](display_resolution) HD display 2 acoustic speaker USB and HDMI port Technos brand
    15.6 ” Led Display Display Screen resolution : 1280 720 (HD Ready) USB : Yes VGS : Yes HDMI : Yes Screen Technology : [led](display_type)
    Model:CG55D1004U
    Screen Size: [55"](display_size)
    Resolution: [3840x2160p](display_resolution)
    Power Supply: 100~240 V/AC
    Sound Output (RMS): 8W + 8W
    Warranty: [3 Years](warranty) wrranty
    """
    return raw_text.split("\n")


from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
def gen_model(model_name):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=37 )#num_labels=37, id2label=dm.id2label, label2id=dm.label2id
    return model, tokenizer

def train():
    is_dry_run=False
    if not is_dry_run:
        raw_text=convert_cprod1_to_rasa()
    else:
        raw_text=gen_temp_data()
    dm = NERDataMaker(raw_text )
    print(f"total examples = {len(dm)}")
    print(dm[0:3])

    # total examples = 35
    # [{'id': 0, 'ner_tags': [0], 'tokens': ['']}, {'id': 1, 'ner_tags': [2, 3, 0],
    model_name="asahi417/tner-xlm-roberta-base-ontonotes5"#"djagatiya/ner-bert-base-cased-ontonotesv5-englishv4"
    model, tokenizer=gen_model(model_name)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    epochs=2
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
    )

    train_ds = dm.as_hf_dataset(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=train_ds, # eval on training set! ONLY for DEMO!!
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    model_dir = f"./results/models_{epochs}"
    trainer.save_model(model_dir  )
    
if __name__ == "__main__":    
    train()