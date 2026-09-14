import random
import re
import numpy as np


def sample_clip_ids(clips, mininum_frames: int, clip_captions=None, skip_caption_set=None):
    def _count_frames(index_list):
        n = 0
        for i in index_list:
            if clip_captions is not None:
                if clip_captions[i] in skip_caption_set:
                    continue
            n += len(clips[i])
        return n

    assert isinstance(clips, list) and isinstance(clips[0], list)
    if clip_captions is not None:
        assert isinstance(clip_captions, list) and (len(clip_captions) == len(clips))
        clip_captions = [c.strip() for c in clip_captions]

    if skip_caption_set is None:
        skip_caption_set = {}

    ids = []
    mid = random.choice(range(len(clips)))
    ids.append(mid)

    left = right = mid

    while (_count_frames(ids) < mininum_frames) and (len(ids) < len(clips)):
        if left - 1 < 0:
            i = right + 1
            right = i
        elif right + 1 >= len(clips):
            i = left - 1
            left = i
        else:
            if random.random() < 0.5:
                i = right + 1
                right = i
            else:
                i = left - 1
                left = i

        ids.append(i)

    ids = sorted(ids)
    if clip_captions is not None:
        ids = [i for i in ids if clip_captions[i] not in skip_caption_set]
    return ids


def sample_frame_ids(num_frames: int, num_segments: int, training: bool):
    nf = num_frames
    ns = num_segments
    out_indices = np.zeros(ns)
    if nf <= ns:
        if training:
            out_indices = np.concatenate((np.arange(nf), np.random.randint(nf, size=ns - nf)), axis=0)
        else:
            out_indices = np.array([(idx % nf) for idx in range(ns)])
        out_indices = np.sort(np.array(out_indices))
    else:
        stride = nf // ns
        strides = np.array([stride] * ns)
        offsets = np.array([1] * (nf - ns * stride) + [0] * (ns * (stride + 1) - nf))
        if training:
            np.random.shuffle(offsets)
        strides += offsets
        cursor = 0
        for idx, each_stride in enumerate(strides):
            left, right = cursor, cursor + each_stride
            cursor += each_stride
            if training:
                out_indices[idx] = np.random.randint(left, right)
            else:
                out_indices[idx] = left

    return [int(i) for i in out_indices]


def pre_caption(caption, max_words):
    caption_raw = caption
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        ' ',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    if not len(caption):
        raise ValueError(f"pre_caption yields invalid text (raw: {caption_raw})")

    return caption
