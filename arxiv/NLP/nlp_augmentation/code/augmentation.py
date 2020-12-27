import numpy as np
import random 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForMaskedLM 
import torch
import compress_fasttext 

class NLP_augmentation():
   

  
  def __init__(self,
               shift_max = 0.4,   # максимальная часть предложения, 
                                  # которая будет адалена при сдвиге
               n_skip = 2,        # количество слов, которое будет 
                                  # удаляться при применнеии пропуска слов              
               max_dist = 0.135,  # максимальное косинусное расстояние между 
                                  # эмбеддингами предлождений               
               mask_n = 4,        # максимальное количество вариантов слов 
                                  # для маскированного слова, предложенное моделью
                                  # для случая, когда маскированию подлежат 
                                  # имеющиеся слова в предложении (синонимы существующих слов)
               mask_n_extend = 5,     # максимальное количество вариантов слов 
                                      # для маскированного слова, предложенное моделью
                                      # для случая, когда маска устанавливается 
                                      # между существующими в предложении словами ( поиск дополнений)
               mask_n_iterations = 0, # количество итераций для поиска дополнений
               mode = 'auto',         # режим работы 'auto' / 'manual' 
               make_shift = True,     # использование сдвига
               make_skip = True,      # использование пропусков
               make_shift_en = True,  # использование сдвига после обратного перевода
               make_skip_en = True,   # использование пропусков после обратного перевода
               make_transl = True,    # использование перевода 
               find_sinonims = True,  # использование масирования для поиска синонимов
               find_add_ons = True,   # использование масирования для поиска дополнений               
               fasttext_comperssed_ru = 'ft_freqprune_100K_20K_pq_100.bin',
               # fasttext эмбедденги слов для подсчёта расстояния между  эмбеддингами предлождений
               # на русском языке
               fasttext_comperssed_en = 'ft_freqprune_en_300.bin'
               # fasttext эмбедденги слов для подсчёта расстояния между  эмбеддингами предлождений
               # на английском языке
               ):


    if mode == 'auto':
      self.make_shift = True
      self.make_skip = True
      self.make_shift_en = True
      self.make_skip_en = True
      self.make_transl = True
      self.find_sinonims = True
      self.find_add_ons = True
    else:
      self.make_shift = make_shift
      self.make_skip = make_skip
      self.make_shift_en = make_shift_en
      self.make_skip_en = make_skip_en
      self.make_transl = make_transl
      self.find_sinonims = find_sinonims
      self.find_add_ons = find_add_ons

    self.mode = mode
    self.shift_max = shift_max     
    self.n_skip = n_skip
    self.max_dist = max_dist
    self.mask_n = mask_n
    self.mask_n_extend = mask_n_extend
    self.mask_n_iterations = mask_n_iterations
    self.gpu_support = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    self.small_model_ru = compress_fasttext.models.CompressedFastTextKeyedVectors.load(fasttext_comperssed_ru)
    
    
    if self.make_transl:
      print('init ru-en translation')

      # fassttext compressed wv
      self.small_model_en = compress_fasttext.models.CompressedFastTextKeyedVectors.load(fasttext_comperssed_en)

      # ru-en translation opus
      self.opus_tok_ru_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")					
      self.opus_ru_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
      self.opus_ru_en.to(self.gpu_support)
      
      print('init en-ru translation')
      # en-ru translation opus
      self.opus_tok_en_ru = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")						
      self.opus_en_ru = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
      self.opus_en_ru.to(self.gpu_support)


      # ru-en translation facebook
      self.fb_tok_ru_en = AutoTokenizer.from_pretrained("facebook/wmt19-ru-en")						
      self.fb_ru_en = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt19-ru-en")
      self.fb_ru_en.to(self.gpu_support)


      # en-ru translation facebook
      self.fb_tok_en_ru = AutoTokenizer.from_pretrained("facebook/wmt19-en-ru")						
      self.fb_en_ru = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt19-en-ru")
      self.fb_en_ru.to(self.gpu_support)

      print('init translation - done')
      
      

    if self.find_sinonims or self.find_add_ons:
      print('init MaskedLM')
      self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
      self.MLmodel = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")
      self.MLmodel.to(self.gpu_support)
      print('init MaskedLM - done')





  def shift_words(self, sentense, max_part = 0.4):
    """
    Генерация новых предложений за счёт сдвига слов в предложении.
    Сдвиг производится на int(max_part * len(sentense)) как слева, так и справа.
    Возвращает список сгенерированных предложений.
    В случае, если int(max_part * len(sentense)) = 0 - shift обработка не производится, 
    возвращается пустой список
    """
    if sentense[-1] == ' ' :
        sentense = sentense[:-1]

    if sentense[0] == ' ' :
        sentense = sentense[1:]

    sent_list = sentense.split(' ')
    len_sent =  len(sent_list)
    shifted_list = []
    shift_words = int(max_part * len(sent_list)) 
   
    if shift_words < 1:
        return shifted_list
    
    i = 0
    shift_left = sent_list.copy().copy()
    shift_right = sent_list.copy().copy()
    while  i < shift_words:
        del shift_left[0]
        shifted_list.append(' '.join(shift_left))
        del shift_right[-1]
        shifted_list.append(' '.join(shift_right))
        i += 1
    return shifted_list



  def shift_skip(self, sentense, n_skip = 2):
    """
    Генерация новых предложений за счёт пропуска слов в предложении.
    Пропуск слов производится последовательно со всем словами в предложении. 
    При n_skip > 1 производится рекурсиваная обработка - при удалении одного слова, 
    применяется алгоритм для последовательного удаления n_skip - 1 слов.
    Возвращает список сгенерированных предложений.
    В случае, если len(sentense) -n_skip < 3  - skip обработка не производится, 
    возвращается пустой список 
    """
    if sentense[-1] == ' ' :
        sentense = sentense[:-1]

    if sentense[0] == ' ' :
        sentense = sentense[1:]
    sent_list = sentense.split(' ')
    len_sent =  len(sent_list)
    shifted_list = []
    
    if len(sent_list) - n_skip < 3:
        return shifted_list

    if n_skip == 1 :     
        for i in range(len_sent):
            shift_tmp = sent_list.copy().copy()
            del shift_tmp[i]
            shifted_list.append(' '.join(shift_tmp))
        return shifted_list
    else:
        for i in range(len_sent):
            shift_tmp = sent_list.copy().copy()
            del shift_tmp[i]
            shifted_list.append(' '.join(shift_tmp))
            shifted_list.extend(self.shift_skip(' '.join(shift_tmp), 
                                                n_skip = n_skip -1))
        return list(set(shifted_list))

  def ru_en_translation(self, sentense, model_tr = 'fb'):
    """
    перевод предложения с русского на ангийский язык.
    Используются модели Facebook (model_tr = 'fb') 
    и OPUS  (model_tr == 'opus').
    Возвращает сроку на  английском языке.
    """
    if model_tr == 'fb':
      inputs = self.fb_tok_ru_en.encode(sentense, return_tensors="pt").to(self.gpu_support) 
      # self.fb_ru_en.to(self.gpu_support)
      outputs = self.fb_ru_en.generate(inputs.to(self.gpu_support), 
                                       max_length=40, 
                                       num_beams=4, 
                                       early_stopping=True).to(self.gpu_support)
      return self.fb_tok_ru_en.decode(outputs[0][1:-1])
    elif model_tr == 'opus':
      inputs = self.opus_tok_ru_en.encode(sentense, return_tensors="pt").to(self.gpu_support) 
      # self.opus_ru_en.to(self.gpu_support)
      outputs = self.opus_ru_en.generate(inputs.to(self.gpu_support), 
                                         max_length=40, 
                                         num_beams=4, 
                                         early_stopping=True).to(self.gpu_support)
      return self.opus_tok_ru_en.decode(outputs[0][1:])
    else:
      return sentense



  def en_ru_translation(self, sentense, model_tr = 'fb'):
    """
    перевод предложения с английского на русский язык.
    Используются модели Facebook (model_tr = 'fb') 
    и OPUS  (model_tr == 'opus').
    Возвращает сроку на русском языке.
    """
    
    if model_tr == 'fb':
      inputs = self.fb_tok_en_ru.encode(sentense, return_tensors="pt").to(self.gpu_support)
      # self.fb_en_ru.to(self.gpu_support)
      outputs = self.fb_en_ru.generate(inputs.to(self.gpu_support), 
                                       max_length=40, 
                                       num_beams=4, 
                                       early_stopping=True)
      return self.fb_tok_en_ru.decode(outputs[0][1:-1])
    
    elif model_tr == 'opus':
      inputs = self.opus_tok_en_ru.encode(sentense, return_tensors="pt").to(self.gpu_support)
      self.opus_en_ru.to(self.gpu_support)
      outputs = self.opus_en_ru.generate(inputs.to(self.gpu_support), 
                                         max_length=40, 
                                         num_beams=4, 
                                         early_stopping=True)
      return self.opus_tok_en_ru.decode(outputs[0][1:])
    
    else:
      return sentense


  def MLM_gen(self, 
              sentense,             # предложение для обработки
              key_vect,             # вектор оригинального предложения
              n_results = 2,        # количество вариантов, предлагаемых моделью 
                                    # на место в  sentense, обозначенное как mask
              in_mask = False,      # использовать поиск дополнений
              max_combinations = 5, # максимальное количество размещений маски в 
                                    # предложении
              mask = '<mask>',      # вид маскирующего элемента
              ):
    """
    генерация текстов за счёт предсказания маскированного слова. 
    Рабочий язык - русский. Если in_mask = True - маска вставляется 
    последовательно между существующими словами в предложении, 
    в противном случае маска последовательно маскирует слова в предложении.
    n_results огранивает количество вариантов для вставки на маскированное 
    место в предложении. В случае, если косинусное расстояние вектора 
    сгенерированного предложения больше self.max_dist - то такое предложение 
    не добавляется в список аугментированных.
    На выходе список сгенерированных предложений.
    """
    aug_list = []
    sentense_list = sentense.split() 
    sentense_len = len(sentense_list)
    if  sentense_len < max_combinations :
      combinations = np.arange(0, sentense_len)
    else:
      combinations = random.sample(range(sentense_len), k=max_combinations-1)
    if in_mask:
      for i in combinations:
        masked_sentense = sentense
        sentense_list = masked_sentense.split()
        sentense_list.insert(i, mask)
        masked_str = ' '.join(sentense_list)
        inputs = self.tokenizer(masked_str, return_tensors="pt").to(self.gpu_support)
        outputs = self.MLmodel(**inputs.to(self.gpu_support))
        results = self.tokenizer.convert_ids_to_tokens(torch.topk(outputs.logits[0, i+1, :],
                                                                  n_results).indices)

        results = [word[1:] if  word[0] == '▁' else word  for word  in results]
        results = [word if  word[0] != '<' else ''  for word  in results]
        results = [word if  len(word) > 2  else ''  for word  in results]


        for word in results:
          del sentense_list[i]
          sentense_list.insert(i, word)
          aug_sent = " ".join(sentense_list)
          if self.cos_dist(key_vect, 
                           self.small_model_ru[aug_sent]/len(aug_sent.split())
                          #  self.seq_embed(aug_sent, lang = 'en')
                           ) < self.max_dist:

                           aug_list.append(aug_sent)
        del sentense_list[i]
    else:
      for i in combinations:
        masked_sentense = sentense
        sentense_list = masked_sentense.split()
        del sentense_list[i]
        sentense_list.insert(i, mask)
        masked_str = ' '.join(sentense_list)
        inputs = self.tokenizer(masked_str, return_tensors="pt").to(self.gpu_support)
        outputs = self.MLmodel(**inputs.to(self.gpu_support))
        results = self.tokenizer.convert_ids_to_tokens(torch.topk(outputs.logits[0, i+1, :],
                                                                  n_results).indices)
        results = [word[1:] if  word[0] == '▁' else word  for word  in results]
        results = [word if  len(word) > 0 and word[0] != '<' else ''  for word  in results]
        results = [word if  len(word) > 2  else ''  for word  in results]
        results = list(set(results))
 
        for word in results:
          del sentense_list[i]
          sentense_list.insert(i, word)
          aug_sent = " ".join(sentense_list)
          if self.cos_dist(key_vect, 
                           self.small_model_ru[aug_sent]/len(aug_sent.split())
                          #  self.seq_embed(aug_sent, lang = 'en')
                           ) < self.max_dist:
                           aug_list.append(aug_sent)
    return aug_list



  def MLM_gen_list(self, 
                   list_sent, 
                   key_wv, 
                   insert_mask = False,
                   n_res = 2,
                   n_comb = 5,
                   mask = '<mask>'                   
                   ):
    masked_sent = []
    for sent in list_sent:
      masked_sent.extend(self.MLM_gen(sentense = sent,
                                      key_vect = key_wv,
                                      n_results  = n_res,
                                      max_combinations = n_comb,
                                      mask = '<mask>' 
                                      )
      )
    return masked_sent



  def cos_dist(self, x,y):
    """
    подсчёт косинусного расстояния между векторами x,y
    возвращает скалярное значение [0;1]
    """
    return 1 - np.inner(x,y)/np.sqrt(np.dot(x,x)*np.dot(y,y))



  def check_similarity(self, sent_list, key_wvect):
    """
    проверяет аугментированные предложения на косинусную близость эмбеддинга предложения
    к   эмбеддингу оригинала. Эмбеддинг предложения находится как сумма эмбеддингов слов (baseline). 
    в случае превышения порогового значения self.max_dist - предложение выбрасывается из списка эмбеддингов
    Возвращает список аугментированных предложений, косинусное расстояние  эбеддингов которых менее self.max_dist 
    """
    del_sentenses = []
    for val in sent_list:
      val_wv = self.small_model_ru[val]/len(val.split())
      # self.seq_embed(val)
      if self.cos_dist(key_wvect, val_wv) > self.max_dist:
        del_sentenses.append(val)
    return list(set(sent_list).difference(set(del_sentenses)))



  def check_end(self, sent_list, augmented_history, original, ru_sent, ru_sent_wv, target_n):
    """
    проверка аугментаций на сходство с оригинальным предложением и на общее 
    количество аугментаций
    Вход:
    sent_list - список аугментаций, 
    augmented_history - созданыне на предыдущих итерациях аугментации, 
    original - флаг возврата оригинального текста в списке аугментаций,
    ru_sent_wv - вектор оригинального текста,
    target_n - целевое количество аугментаций,
    Выход:
    aug_complete - bool значение о достаточности количества аугментаций
    sent_list - проверенный список аугментаций на близость к оригинальному
    предложению
    """
    sent_list = list(set(sent_list))
    sent_list = self.check_similarity(sent_list, ru_sent_wv)
    setA = set(sent_list)
    setB = set(augmented_history)
    sent_list = list(setA.difference(setB))
    if not original and ru_sent in sent_list:
      sent_list.remove(ru_sent)
    if ( self.mode  not in  ['auto', 'extend']):
      return True, sent_list[:target_n]
    elif (len(sent_list) >  target_n -1):
      return True, sent_list[:target_n]
    else:
      return False, sent_list

  def aug_pipline(self, 
                  sent,                   # предложение для обработки  
                  augmented_list = [],    # список уже сохданных аугментаций              
                  add_original = False,   # добавление 
                  n_aug = 500,            # требуемое количество аугментаций
                  ):
    """
    построение линии аугментации текстового предложения.
    входные параметры:
    sent - предложение для угмеентации (строка)
    mode = режим работы ('auto', 'manual', default = 'auto')
    add_original  - опция добавления оригинального предложения 
    в список выдачи(default = False)
    n_aug  - требуемое количество аугментаций 
    в случае режима работы 'auto' (int, default = 100)
    выход:
    список строк, содаржащий аугментированные данные.  
    """
    if self.mode == 'auto':
      self.make_shift = True
      self.make_skip = True
      self.make_shift_en = True
      self.make_skip_en = True
      self.make_transl = True
      self.find_sinonims = True
      self.find_add_ons = True
    elif  self.mode == 'extend':
      self.make_shift = False
      self.make_skip = False
      self.make_shift_en = True
      self.make_skip_en = True
      self.make_transl = True
      self.find_sinonims = False
      self.find_add_ons = True



    ru_sent_wv = self.small_model_ru[sent]/len(sent.split())
    len_sent = len(sent.split())
    augmented_sent = []
    augmented_sent.append(sent)

    # применение сдвига
    if self.make_shift:
      augmented_sent.extend(self.shift_words(sent,self.shift_max))

    # применение удаления слов    
    if self.make_skip:
      augmented_sent.extend(self.shift_skip(sent, self.n_skip))


    if self.make_skip or self.make_shift:
      aug_complete, augmented_sent = self.check_end(sent_list = augmented_sent, 
                                                    augmented_history = augmented_list, 
                                                    original = add_original, 
                                                    ru_sent = sent, 
                                                    ru_sent_wv = ru_sent_wv, 
                                                    target_n = n_aug)
      if aug_complete:
        return augmented_sent

    # поиск синонимов с помощью MaskedLM
    if self.find_sinonims and len_sent > 5:
      augmented_sent.extend(self.MLM_gen_list(list_sent = augmented_sent,
                                              key_wv = ru_sent_wv,
                                              n_res = self.mask_n)
      )
      aug_complete, augmented_sent = self.check_end(sent_list = augmented_sent, 
                                                    augmented_history = augmented_list, 
                                                    original = add_original, 
                                                    ru_sent = sent, 
                                                    ru_sent_wv = ru_sent_wv, 
                                                    target_n = n_aug)
      if aug_complete:
        return augmented_sent



    # поиск дополнений
    if self.find_add_ons:
      extended_mask = []
      for i in range(self.mask_n_iterations+1):
        if len(extended_mask) == 0:
          extended_mask.extend(self.MLM_gen_list(list_sent = augmented_sent, 
                                                 key_wv = ru_sent_wv,
                                                 insert_mask =True,
                                                 n_res = self.mask_n_extend 
                                                 )
          )
        else:
          extended_mask.extend(self.MLM_gen_list(list_sent = extended_mask, 
                                                 key_wv = ru_sent_wv,
                                                 insert_mask =True,
                                                 n_res = self.mask_n_extend
                                                 )
          )



      augmented_sent.extend(extended_mask)
      aug_complete, augmented_sent = self.check_end(sent_list = augmented_sent, 
                                                    augmented_history = augmented_list, 
                                                    original = add_original, 
                                                    ru_sent = sent, 
                                                    ru_sent_wv = ru_sent_wv, 
                                                    target_n = n_aug)
      if aug_complete:
        return augmented_sent


    # обратноый перевод
    if self.make_transl:

      en_sent = self.ru_en_translation(sent)
      en_sent_wv = self.small_model_en[en_sent]/len(en_sent.split()) 


      augmented_sent_en = []
      for seq in augmented_sent:

        augmented_sent_en.append(self.ru_en_translation(seq))
        augmented_sent_en.append(self.ru_en_translation(seq, model_tr = 'opus'))
      augmented_sent_en = list(set(augmented_sent_en))

      for seq in augmented_sent_en:
        augmented_sent.append(self.en_ru_translation(seq))
        augmented_sent.append(self.en_ru_translation(seq, model_tr = 'opus'))

      aug_complete, augmented_sent = self.check_end(sent_list = augmented_sent, 
                                                    augmented_history = augmented_list, 
                                                    original = add_original, 
                                                    ru_sent = sent, 
                                                    ru_sent_wv = ru_sent_wv, 
                                                    target_n = n_aug)
      if aug_complete:
        return augmented_sent

      # сдвиг и удаление слов после обратного перевода
      add_augmented_sent = []
      for setense in augmented_sent:
        if self.make_shift_en:
          add_augmented_sent.extend(self.shift_words(setense, self.shift_max))
        if self.make_skip_en:
          add_augmented_sent.extend(self.shift_skip(setense, self.n_skip)) 
      augmented_sent.extend(add_augmented_sent)
      aug_complete, augmented_sent = self.check_end(sent_list = augmented_sent, 
                                                    augmented_history = augmented_list, 
                                                    original = add_original, 
                                                    ru_sent = sent, 
                                                    ru_sent_wv = ru_sent_wv, 
                                                    target_n = n_aug)
      if aug_complete:
        return list(set(augmented_sent ))
    
    if ( self.mode  not in  ['auto', 'extend']):
      return list(set(augmented_sent ))

    # случай, когда алгоритму не удалось набрать требуемое количество аугментаций
    # при начальных условиях 
    if self.mask_n_iterations > 15:
      if add_original:
        augmented_sent.append(sent)
        self.mask_n_iterations = 1
        return list(set(augmented_sent ))
    if sent in augmented_sent:
      augmented_sent.remove(sent)
    if (len(augmented_sent) <  n_aug):
      self.mask_n_iterations = +1
      extended_sent = max(augmented_sent, key = len)
      if self.mode == 'auto':
        extended_sent = sent
      elif len([word for word in extended_sent.split() if len(word) > 1]) < len_sent:
        extended_sent = sent
      elif extended_sent in augmented_sent:
        augmented_sent.remove(extended_sent)
      augmented_sent.extend(augmented_list)


      self.mode = 'extend'
      add_aug = self.aug_pipline(extended_sent, 
                                 add_original = add_original,
                                 augmented_list = augmented_sent,
                                 n_aug = n_aug - len(augmented_sent))
      augmented_sent.extend(add_aug)

      if add_original:
        augmented_sent.append(sent)
        self.mask_n_iterations = 1
        return list(set(augmented_sent ))
      else:
        self.mask_n_iterations = 1
        return list(set(augmented_sent ))
    return list(set(augmented_sent ))
    


