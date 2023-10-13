GIRL = '1girl'
BOY = '1boy'
MAN = '1man'
WOMAN = '1woman'
LITTLE_GIRL = 'a little girl'
YOUNG_MAN = 'a young man'

MERGE_MAP = {
    GIRL: '2 girls',
    BOY: '2 boys',
    MAN: '2 men',
    WOMAN: '2 women',
    LITTLE_GIRL: '2 little girls',
    YOUNG_MAN: '2 young men',
}

LABEL_MAP = {
    GIRL: 'Female',
    BOY: 'Male',
    MAN: 'Male',
    WOMAN: 'Female',
    LITTLE_GIRL: 'Female',
    YOUNG_MAN: 'Male',
}

class GenderPrompt:
    @staticmethod
    def merge(gender1, gender2, weight):
        if gender1 == gender2:
            return f'({MERGE_MAP[gender1]}:{weight})'
        else:
            return f'({gender1} and {gender2}:{weight})'
