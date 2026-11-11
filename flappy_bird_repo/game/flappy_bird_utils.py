import pygame
import sys
import os

def load():
    base_path = os.path.dirname(os.path.dirname(__file__))
    assets_path = os.path.join(base_path, 'assets')
    sprites_path = os.path.join(assets_path, 'sprites')
    audio_path = os.path.join(assets_path, 'audio')

    # path of player with different states
    PLAYER_PATH = (
            os.path.join(sprites_path, 'redbird-upflap.png'),
            os.path.join(sprites_path, 'redbird-midflap.png'),
            os.path.join(sprites_path, 'redbird-downflap.png')
    )

    # path of background
    BACKGROUND_PATH = os.path.join(sprites_path, 'background-black.png')

    # path of pipe
    PIPE_PATH = os.path.join(sprites_path, 'pipe-green.png')

    IMAGES, SOUNDS, HITMASKS = {}, {}, {}

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load(os.path.join(sprites_path, '0.png')).convert_alpha(),
        pygame.image.load(os.path.join(sprites_path, '1.png')).convert_alpha(),
        pygame.image.load(os.path.join(sprites_path, '2.png')).convert_alpha(),
        pygame.image.load(os.path.join(sprites_path, '3.png')).convert_alpha(),
        pygame.image.load(os.path.join(sprites_path, '4.png')).convert_alpha(),
        pygame.image.load(os.path.join(sprites_path, '5.png')).convert_alpha(),
        pygame.image.load(os.path.join(sprites_path, '6.png')).convert_alpha(),
        pygame.image.load(os.path.join(sprites_path, '7.png')).convert_alpha(),
        pygame.image.load(os.path.join(sprites_path, '8.png')).convert_alpha(),
        pygame.image.load(os.path.join(sprites_path, '9.png')).convert_alpha()
    )

    # base (ground) sprite
    IMAGES['base'] = pygame.image.load(os.path.join(sprites_path, 'base.png')).convert_alpha()

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    SOUNDS['die']    = pygame.mixer.Sound(os.path.join(audio_path, 'die' + soundExt))
    SOUNDS['hit']    = pygame.mixer.Sound(os.path.join(audio_path, 'hit' + soundExt))
    SOUNDS['point']  = pygame.mixer.Sound(os.path.join(audio_path, 'point' + soundExt))
    SOUNDS['swoosh'] = pygame.mixer.Sound(os.path.join(audio_path, 'swoosh' + soundExt))
    SOUNDS['wing']   = pygame.mixer.Sound(os.path.join(audio_path, 'wing' + soundExt))

    # select random background sprites
    IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()

    # select random player sprites
    IMAGES['player'] = (
        pygame.image.load(PLAYER_PATH[0]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[1]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[2]).convert_alpha(),
    )

    # select random pipe sprites
    IMAGES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load(PIPE_PATH).convert_alpha(), 180),
        pygame.image.load(PIPE_PATH).convert_alpha(),
    )

    # hismask for pipes
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    # hitmask for player
    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )

    return IMAGES, SOUNDS, HITMASKS

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask
