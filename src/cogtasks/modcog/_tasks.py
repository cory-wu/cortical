"""Mod-Cog task factories (Khona, Chandra et al. 2023), vendored verbatim.

Source: https://github.com/mikailkhona/Mod_Cog (mod_cog_tasks.py, commit
41101436a130fc31deb61e81cc48925c75bed617, CC0-1.0). Function bodies are
unchanged, including upstream quirks (e.g. ``dlyantiseqr`` overwrites its
``delay`` timing). The 11 upstream ``*intseq`` factories, which are outside
both the Cog20 and the 82-task Mod-Cog sets, are removed.
"""

from neurogym.utils import scheduler
from neurogym.wrappers.block import ScheduleEnvs

from ._envs import _DelayMatch1DResponse, _DMFamily, _MultiModalityStimulus, _Reach

#First family of tasks (Go family)
def _reach(**kwargs):
    envs = list()
    for modality in [0, 1]:
        env = _Reach(**kwargs)
        env = _MultiModalityStimulus(env, modality=modality, n_modality=2)
        envs.append(env)
    schedule = scheduler.RandomSchedule(len(envs))
    env = ScheduleEnvs(envs, schedule, env_input=False)
    return env

def go(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = False
    return _reach(**env_kwargs)

def goseqr(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = False
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = 1
    return _reach(**env_kwargs)

def goseql(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = False
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = -1
    return _reach(**env_kwargs)

def anti(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = True
    return _reach(**env_kwargs)

def antiseqr(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = True
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = 1
    return _reach(**env_kwargs)

def antiseql(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = True
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = -1
    return _reach(**env_kwargs)


def rtgo(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = False
    env_kwargs['reaction'] = True
    return _reach(**env_kwargs)


def rtgoseqr(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = False
    env_kwargs['reaction'] = True
    env_kwargs['seq'] = 1
    return _reach(**env_kwargs)

def rtgoseql(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = False
    env_kwargs['reaction'] = True
    env_kwargs['seq'] = -1
    return _reach(**env_kwargs)

def rtanti(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = True
    env_kwargs['reaction'] = True
    return _reach(**env_kwargs)

def rtantiseqr(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = True
    env_kwargs['reaction'] = True
    env_kwargs['seq'] = 1
    return _reach(**env_kwargs)

def rtantiseql(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = True
    env_kwargs['reaction'] = True
    env_kwargs['seq'] = -1
    return _reach(**env_kwargs)
    

def dlygo(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = False
    env_kwargs['dly'] = True
    return _reach(**env_kwargs)

def dlygoseqr(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = False
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = 1
    return _reach(**env_kwargs)

def dlygoseql(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = False
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = -1
    return _reach(**env_kwargs)


def dlygointr(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = False
    env_kwargs['integrate'] = 1
    env_kwargs['dly'] = True
    return _reach(**env_kwargs)

def dlygointl(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = False
    env_kwargs['integrate'] = -1
    env_kwargs['dly'] = True
    return _reach(**env_kwargs)

def dlyanti(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = True
    env_kwargs['timing'] = {'delay': 500}
    return _reach(**env_kwargs)

def dlyantiseqr(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = True
    env_kwargs['timing'] = {'delay': 500}
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = 1
    return _reach(**env_kwargs)

def dlyantiseql(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = True
    env_kwargs['timing'] = {'delay': 500}
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = -1
    return _reach(**env_kwargs)

def dlyantiintr(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = True
    env_kwargs['integrate'] = 1
    #env_kwargs['timing'] = {'delay': 500}
    return _reach(**env_kwargs)

def dlyantiintl(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = True
    env_kwargs['integrate'] = -1
    #env_kwargs['timing'] = {'delay': 500}
    return _reach(**env_kwargs)

#Second family of tasks (DM Family)

def _dm_kwargs():
    env_kwargs = {'cohs': [0.08, 0.16, 0.32, 0.64],
                  'delaycomparison': False}
    return env_kwargs


def dm1(**kwargs):
    env_kwargs = _dm_kwargs()
    env_kwargs.update({'w_mod': (1, 1), 'stim_mod': (True, False)})
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)


def dm1seqr(**kwargs):
    env_kwargs = _dm_kwargs()
    env_kwargs.update({'w_mod': (1, 1), 'stim_mod': (True, False)})
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = 1
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)

def dm1seql(**kwargs):
    env_kwargs = _dm_kwargs()
    env_kwargs.update({'w_mod': (1, 1), 'stim_mod': (True, False)})
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = -1
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)


def dm2(**kwargs):
    env_kwargs = _dm_kwargs()
    env_kwargs.update({'w_mod': (1, 1), 'stim_mod': (False, True)})
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)

def dm2seqr(**kwargs):
    env_kwargs = _dm_kwargs()
    env_kwargs.update({'w_mod': (1, 1), 'stim_mod': (False, True)})
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = 1
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)

def dm2seql(**kwargs):
    env_kwargs = _dm_kwargs()
    env_kwargs.update({'w_mod': (1, 1), 'stim_mod': (False, True)})
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = -1
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)

def ctxdm1(**kwargs):
    env_kwargs = _dm_kwargs()
    env_kwargs.update({'w_mod': (1, 0), 'stim_mod': (True, True)})
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)

def ctxdm1seqr(**kwargs):
    env_kwargs = _dm_kwargs()
    env_kwargs.update({'w_mod': (1, 0), 'stim_mod': (True, True)})
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = 1
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)

def ctxdm1seql(**kwargs):
    env_kwargs = _dm_kwargs()
    env_kwargs.update({'w_mod': (1, 0), 'stim_mod': (True, True)})
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = -1
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)

def ctxdm2(**kwargs):
    env_kwargs = _dm_kwargs()
    env_kwargs.update({'w_mod': (0, 1), 'stim_mod': (True, True)})
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)

def ctxdm2seqr(**kwargs):
    env_kwargs = _dm_kwargs()
    env_kwargs.update({'w_mod': (0, 1), 'stim_mod': (True, True)})
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = 1
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)

def ctxdm2seql(**kwargs):
    env_kwargs = _dm_kwargs()
    env_kwargs.update({'w_mod': (0, 1), 'stim_mod': (True, True)})
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = -1
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)

def multidm(**kwargs):
    env_kwargs = _dm_kwargs()
    env_kwargs.update({'w_mod': (1, 1), 'stim_mod': (True, True)})
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)


def multidmseqr(**kwargs):
    env_kwargs = _dm_kwargs()
    env_kwargs.update({'w_mod': (1, 1), 'stim_mod': (True, True)})
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = 1
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)

def multidmseql(**kwargs):
    env_kwargs = _dm_kwargs()
    env_kwargs.update({'w_mod': (1, 1), 'stim_mod': (True, True)})
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = -1
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)

#Third family: DM family with Delay

def _dlydm_kwargs():
    env_kwargs = {'cohs': [0.3, 0.6, 1.0],
                  'delaycomparison': True}
    return env_kwargs


def dlydm1(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (1, 1), 'stim_mod': (True, False)})
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)

def dlydm1seqr(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (1, 1), 'stim_mod': (True, False)})
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = 1
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)

def dlydm1seql(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (1, 1), 'stim_mod': (True, False)})
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = -1
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)

def dlydm1intr(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (1, 1), 'stim_mod': (True, False)})
    env_kwargs.update(kwargs)
    env_kwargs['integrate'] = 1
    return _DMFamily(**env_kwargs)

def dlydm1intl(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (1, 1), 'stim_mod': (True, False)})
    env_kwargs.update(kwargs)
    env_kwargs['integrate'] = -1
    return _DMFamily(**env_kwargs)


def dlydm2(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (1, 1), 'stim_mod': (False, True)})
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)

def dlydm2seqr(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (1, 1), 'stim_mod': (False, True)})
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = 1
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)

def dlydm2seql(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (1, 1), 'stim_mod': (False, True)})
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = -1
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)

def dlydm2intr(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (1, 1), 'stim_mod': (False, True)})
    env_kwargs.update(kwargs)
    env_kwargs['integrate'] = 1
    return _DMFamily(**env_kwargs)

def dlydm2intl(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (1, 1), 'stim_mod': (False, True)})
    env_kwargs.update(kwargs)
    env_kwargs['integrate'] = -1
    return _DMFamily(**env_kwargs)

def ctxdlydm1(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (1, 0), 'stim_mod': (True, True)})
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)

def ctxdlydm1seqr(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (1, 0), 'stim_mod': (True, True)})
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = 1
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)

def ctxdlydm1seql(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (1, 0), 'stim_mod': (True, True)})
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = -1
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)


def ctxdlydm1intr(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (1, 0), 'stim_mod': (True, True)})
    env_kwargs.update(kwargs)
    env_kwargs['integrate'] = 1
    return _DMFamily(**env_kwargs)

def ctxdlydm1intl(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (1, 0), 'stim_mod': (True, True)})
    env_kwargs.update(kwargs)
    env_kwargs['integrate'] = -1
    return _DMFamily(**env_kwargs)

def ctxdlydm2(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (0, 1), 'stim_mod': (True, True)})
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)


def ctxdlydm2seqr(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (0, 1), 'stim_mod': (True, True)})
    env_kwargs.update(kwargs)
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = 1
    return _DMFamily(**env_kwargs)


def ctxdlydm2seql(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (0, 1), 'stim_mod': (True, True)})
    env_kwargs.update(kwargs)
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = -1
    return _DMFamily(**env_kwargs)

def ctxdlydm2intr(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (0, 1), 'stim_mod': (True, True)})
    env_kwargs.update(kwargs)
    env_kwargs['integrate'] = 1
    return _DMFamily(**env_kwargs)

def ctxdlydm2intl(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (0, 1), 'stim_mod': (True, True)})
    env_kwargs.update(kwargs)
    env_kwargs['integrate'] = -1
    return _DMFamily(**env_kwargs)

def multidlydm(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (1, 1), 'stim_mod': (True, True)})
    env_kwargs.update(kwargs)
    return _DMFamily(**env_kwargs)

def multidlydmseqr(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (1, 1), 'stim_mod': (True, True)})
    env_kwargs.update(kwargs)
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = 1
    return _DMFamily(**env_kwargs)

def multidlydmseql(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (1, 1), 'stim_mod': (True, True)})
    env_kwargs.update(kwargs)
    env_kwargs['timing'] = {'decision': 1000}
    env_kwargs['seq'] = -1
    return _DMFamily(**env_kwargs)

def multidlydmintr(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (1, 1), 'stim_mod': (True, True)})
    env_kwargs.update(kwargs)
    env_kwargs['integrate'] = 1
    return _DMFamily(**env_kwargs)

def multidlydmintl(**kwargs):
    env_kwargs = _dlydm_kwargs()
    env_kwargs.update({'w_mod': (1, 1), 'stim_mod': (True, True)})
    env_kwargs.update(kwargs)
    env_kwargs['integrate'] = -1
    return _DMFamily(**env_kwargs)

#Delay + match family
def _dlymatch(matchto, matchgo, **kwargs):
    envs = list()
    for modality in [0, 1]:
        env_kwargs = {'matchto': matchto, 'matchgo': matchgo}
        env_kwargs.update(kwargs)
        env = _DelayMatch1DResponse(**env_kwargs)
        env = _MultiModalityStimulus(env, modality=modality, n_modality=2)
        envs.append(env)
    schedule = scheduler.RandomSchedule(len(envs))
    env = ScheduleEnvs(envs, schedule, env_input=False)
    return env


def dms(**kwargs):
    return _dlymatch(matchto='sample', matchgo=True, **kwargs)

def dmsintr(**kwargs):
    return _dlymatch(matchto='sample', matchgo=True, integrate=1, **kwargs)

def dmsintl(**kwargs):
    return _dlymatch(matchto='sample', matchgo=True, integrate=-1, **kwargs)

def dnms(**kwargs):
    return _dlymatch(matchto='sample', matchgo=False, **kwargs)

def dnmsintr(**kwargs):
    return _dlymatch(matchto='sample', matchgo=False, integrate=1, **kwargs)

def dnmsintl(**kwargs):
    return _dlymatch(matchto='sample', matchgo=False, integrate=-1, **kwargs)

def dmc(**kwargs):
    return _dlymatch(matchto='category', matchgo=True, **kwargs)

def dmcintr(**kwargs):
    return _dlymatch(matchto='category', matchgo=True, integrate=1, **kwargs)

def dmcintl(**kwargs):
    return _dlymatch(matchto='category', matchgo=True, integrate=-1, **kwargs)

def dnmc(**kwargs):
    return _dlymatch(matchto='category', matchgo=False, **kwargs)

def dnmcintr(**kwargs):
    return _dlymatch(matchto='category', matchgo=False, integrate=1, **kwargs)

def dnmcintl(**kwargs):
    return _dlymatch(matchto='category', matchgo=False, integrate=-1, **kwargs)

#seq
def dmsseqr(**kwargs):
    return _dlymatch(matchto='sample', matchgo=True, seq=1, timing={'decision': 1000}, **kwargs)

def dmsseql(**kwargs):
    return _dlymatch(matchto='sample', matchgo=True, seq=-1, timing={'decision': 1000}, **kwargs)

def dnmsseqr(**kwargs):
    return _dlymatch(matchto='sample', matchgo=False, seq=1, timing={'decision': 1000}, **kwargs)

def dnmsseql(**kwargs):
    return _dlymatch(matchto='sample', matchgo=False, seq=-1, timing={'decision': 1000}, **kwargs)

def dmcseqr(**kwargs):
    return _dlymatch(matchto='category', matchgo=True, seq=1, timing={'decision': 1000}, **kwargs)

def dmcseql(**kwargs):
    return _dlymatch(matchto='category', matchgo=True, seq=-1, timing={'decision': 1000}, **kwargs)

def dnmcseqr(**kwargs):
    return _dlymatch(matchto='category', matchgo=False, seq=1, timing={'decision': 1000}, **kwargs)

def dnmcseql(**kwargs):
    return _dlymatch(matchto='category', matchgo=False, seq=-1, timing={'decision': 1000}, **kwargs)

