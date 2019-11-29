import json
import os
import os.path
import re

from PIL import Image
import h5py
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import config
import utils
import pickle


def get_loader(image_path, train=False, val=False, test=False):
    """ Returns a data loader for the desired split """
    assert train + val + test == 1, 'need to set exactly one of {train, val, test} to True'
    split = VQA(
        utils.path_for(train=train, val=val, test=test, question=True),
        utils.path_for(train=train, val=val, test=test, answer=True),
        image_path,
        answerable_only = train,
        transform = utils.get_transform(config.image_size, config.central_fraction)
    )
    if test:
        loader = torch.utils.data.DataLoader(
            split,
            batch_size=config.test_batch_size,
            shuffle=train,  # only shuffle the data in training
            pin_memory=True,
            num_workers=config.data_workers,
            collate_fn=collate_fn,
        )
    else:   
        loader = torch.utils.data.DataLoader(
            split,
            batch_size=config.batch_size,
            shuffle=train,  # only shuffle the data in training
            pin_memory=True,
            num_workers=config.data_workers,
            collate_fn=collate_fn,
        )
    return loader


def collate_fn(batch):
    # put question lengths in descending order so that we can use packed sequences later
    batch.sort(key=lambda x: x[-1], reverse=True)
    return data.dataloader.default_collate(batch)


class VQA(data.Dataset):
    """ VQA dataset, open-ended """
    def __init__(self, questions_path, answers_path, image_path, answerable_only=False, transform=None):
        super(VQA, self).__init__()
        # print(questions_path)
        with open(questions_path, 'r') as fd:
            questions_json = json.load(fd)
        # print(questions_json)
        with open(answers_path, 'r') as fd:
            answers_json = json.load(fd)
        # print(config.captions_vocabulary_path)
        with open(config.captions_vocabulary_path, 'rb') as fd:
            cap_vocab_json = pickle.load(fd)
        with open(config.hashtags_vocabulary_path, 'rb') as fd:
            hash_vocab_json = pickle.load(fd)
        self._check_integrity(questions_json, answers_json)

        # vocab
        # self.vocab = vocab_json
        self.token_to_index = cap_vocab_json
        self.answer_to_index = hash_vocab_json

        # q and a
        self.questions = list(prepare_questions(questions_json))
        self.answers = list(prepare_answers(answers_json))
        self.questions = [self._encode_question(q) for q in self.questions]
        self.answers = [self._encode_answers(a) for a in self.answers]

        #v
        self.path = image_path
        self.id_to_filename = self._find_images()
        self.sorted_ids = sorted(self.id_to_filename.keys())  # used for deterministic iteration order
        # print('found {} images in {}'.format(len(self), self.path))
        self.transform = transform

        # v
        # self.image_features_path = image_features_path
        # self.coco_id_to_index = self._create_coco_id_to_index()
        # self.coco_ids = [q['image_id'] for q in questions_json['questions']]
        self.coco_ids = [key for key in questions_json.keys()]

        # only use questions that have at least one answer?
        self.answerable_only = answerable_only
        # if self.answerable_only:
        #     self.answerable = self._find_answerable()

    @property
    def max_question_length(self):
        if not hasattr(self, '_max_length_questions'):
            self._max_length_questions = max(map(len, self.questions))
        return self._max_length_questions
    
    @property
    def max_answer_length(self):
        if not hasattr(self, '_max_length_answers'):
            self._max_length_answers = max(map(len, self.answers))
        return self._max_length_answers

    @property
    def num_tokens(self):
        return len(self.token_to_index),len(self.answer_to_index)  # add 1 for <unknown> token at index 0

    def _create_coco_id_to_index(self):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
        with h5py.File(self.image_features_path, 'r') as features_file:
            coco_ids = features_file['ids'][()]
        coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
        return coco_id_to_index

    def _check_integrity(self, questions, answers):
        """ Verify that we are using the correct data """
        # print('2019-11-19_15-30-46_UTC.jpg' in questions.keys())
        flag = True
        for key in questions.keys():
            if key not in answers.keys():
                flag == False
        # qa_pairs = list(zip(questions['questions'], answers['annotations']))
        assert flag == True, 'Questions not aligned with answers'
        # assert all(q['image_id'] == a['image_id'] for q, a in qa_pairs), 'Image id of question and answer don\'t match'
        # assert questions['data_type'] == answers['data_type'], 'Mismatched data types'
        # assert questions['data_subtype'] == answers['data_subtype'], 'Mismatched data subtypes'

    def _find_answerable(self):
        """ Create a list of indices into questions that will have at least one answer that is in the vocab """
        answerable = []
        for i, answers in enumerate(self.answers):
            answer_has_index = len(answers.nonzero()) > 0
            # store the indices of anything that is answerable
            if answer_has_index:
                answerable.append(i)
        return answerable

    def _encode_question(self, question):
        """ Turn a question into a vector of indices and a question length """
        vec = torch.zeros(self.max_question_length + 2).long()
        vec[0] = self.token_to_index['<sos>']
        for i, token in enumerate(question):
            index = self.token_to_index.get(token, 0)
            vec[i+1] = index
        vec[i+2] = self.token_to_index['<eos>']
        return vec, len(question)

    def _encode_answers(self, answers):
        """ Turn an answer into a vector """
        # answer vec will be a vector of answer counts to determine which answers will contribute to the loss.
        # this should be multiplied with 0.1 * negative log-likelihoods that a model produces and then summed up
        # to get the loss that is weighted by how many humans gave that answer
        answer_vec = torch.zeros(self.max_answer_length+2).long()
        # print(answers)
        answer_vec[0] = self.answer_to_index['<sos>']
        for i, token in enumerate(answers):
            index = self.answer_to_index.get(token, 0)
            answer_vec[i+1] = index
        if len(answers) == 0:
            answer_vec[1] = self.answer_to_index['<eos>']
        else:
            answer_vec[i+2] = self.answer_to_index['<eos>']
        return answer_vec, len(answers)
        # for answer in answers:
        #     index = self.answer_to_index.get(answer)
        #     if index is not None:
        #         answer_vec[index] += 1
        # return answer_vec

    def _load_image(self, image_id):
        """ Load an image """
        if not hasattr(self, 'features_file'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.features_file = h5py.File(self.image_features_path, 'r')
        index = self.coco_id_to_index[image_id]
        dataset = self.features_file['features']
        img = dataset[index].astype('float32')
        return torch.from_numpy(img)

    def _find_images(self):
        id_to_filename = {}
        for filename in os.listdir(self.path):
            if not filename.endswith('.jpg'):
                continue
            # print(filename)
            # id_and_extension = filename.split('_')[-1]
            # print(id_and_extension)
            id = filename.split('.')[0]
            id_to_filename[id] = filename
        return id_to_filename

    def __getitem__(self, item):
        # if self.answerable_only:
        #     # change of indices to only address answerable questions
        #     item = self.answerable[item]

        q, q_length = self.questions[item]
        a,a_length = self.answers[item]
        # print(q)
        inv_map = {v: k for k, v in self.token_to_index.items()}
        inv_map_ans = {v: k for k, v in self.answer_to_index.items()}
        # print(self.token_to_index)
        # print(inv_map)
        q_tr = []
        a_tr = []
        # for w in q:
        #     if w == 0:
        #         break
        #     q_tr.append(inv_map[w.item()])
        # print(a)
        # for i,w in enumerate(a):
        #     if w == 1:
        #         a_tr.append(inv_map_ans[i])
        # print(q_tr)
        # print(a_tr)
        image_id = self.coco_ids[item]
        # v = self._load_image(image_id)
        # id = self.sorted_ids[image_id]
        path = os.path.join(self.path, image_id)
        # print(path)
        # klajfd
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        # since batches are re-ordered for PackedSequence's, the original question order is lost
        # we return `item` so that the order of (v, q, a) triples can be restored if desired
        # without shuffling in the dataloader, these will be in the order that they appear in the q and a json's.
        return img, q, a, item, q_length, a_length

    def __len__(self):
        if self.answerable_only:
            return len(self.answers)
        else:
            return len(self.questions)


# this is used for normalizing questions
_special_chars = re.compile('[^a-z0-9 ]*')

# these try to emulate the original normalization scheme for answers
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))


def prepare_questions(questions_json):
    """ Tokenize and normalize questions from a given question json in the usual VQA format. """
    questions = [questions_json[key] for key in questions_json.keys()]
    # questions = [q['question'] for q in questions_json['questions']]

    for question in questions:
        yield re.findall(r"[\w']+|[.,!?;/]", question.lower())
        # question = question.lower()[:-1]
        # yield question.split(' ')


def prepare_answers(answers_json):
    """ Normalize answers from a given answer json in the usual VQA format. """
    # answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in answers_json['annotations']]
    answers = [answers_json[key] for key in answers_json.keys()]
    # for answer in answers:
    #     answer = answer.lower()[:-1]
    #     yield answer.split(' ')
    # The only normalization that is applied to both machine generated answers as well as
    # ground truth answers is replacing most punctuation with space (see [0] and [1]).
    # Since potential machine generated answers are just taken from most common answers, applying the other
    # normalizations is not needed, assuming that the human answers are already normalized.
    # [0]: http://visualqa.org/evaluation.html
    # [1]: https://github.com/VT-vision-lab/VQA/blob/3849b1eae04a0ffd83f56ad6f70ebd0767e09e0f/PythonEvaluationTools/vqaEvaluation/vqaEval.py#L96

    def process_punctuation(s):
        # the original is somewhat broken, so things that look odd here might just be to mimic that behaviour
        # this version should be faster since we use re instead of repeated operations on str's
        if _punctuation.search(s) is None:
            return s
        s = _punctuation_with_a_space.sub('', s)
        if re.search(_comma_strip, s) is not None:
            s = s.replace(',', '')
        s = _punctuation.sub(' ', s)
        s = _period_strip.sub('', s)
        return s.strip()

    for answer in answers:
        yield re.findall(r"[\w']+|[.,!?;/]", answer.lower())
    # for answer_list in answers:
    #     yield list(map(process_punctuation, answer_list))


class InstaImages(data.Dataset):
    """ Dataset for MSCOCO images located in a folder on the filesystem """
    def __init__(self, path, transform=None):
        super(InstaImages, self).__init__()
        self.path = path
        self.id_to_filename = self._find_images()
        self.sorted_ids = sorted(self.id_to_filename.keys())  # used for deterministic iteration order
        print('found {} images in {}'.format(len(self), self.path))
        self.transform = transform

    def _find_images(self):
        id_to_filename = {}
        for filename in os.listdir(self.path):
            if not filename.endswith('.jpg'):
                continue
            print(filename)
            # id_and_extension = filename.split('_')[-1]
            # print(id_and_extension)
            id = filename.split('.')[0]
            id_to_filename[id] = filename
        return id_to_filename

    def __getitem__(self, item):
        id = self.sorted_ids[item]
        path = os.path.join(self.path, self.id_to_filename[id])
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return id, img

    def __len__(self):
        return len(self.sorted_ids)


class Composite(data.Dataset):
    """ Dataset that is a composite of several Dataset objects. Useful for combining splits of a dataset. """
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, item):
        current = self.datasets[0]
        for d in self.datasets:
            if item < len(d):
                return d[item]
            item -= len(d)
        else:
            raise IndexError('Index too large for composite dataset')

    def __len__(self):
        return sum(map(len, self.datasets))
