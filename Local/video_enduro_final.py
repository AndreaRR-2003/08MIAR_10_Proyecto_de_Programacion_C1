# generar_video_enduro.py

import os
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
from PIL import Image
import gym
from gym import wrappers
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Permute, Input

try:
    from tensorflow.keras.optimizers.legacy import Adam
except Exception:
    from tensorflow.keras.optimizers import Adam

keras.__version__ = tf.__version__

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor


# =========================
# CONFIGURACIÓN
# =========================

PROJECT_ROOT = r"."  # carpeta donde están los pesos
ENV_NAME = "Enduro-v0"


VIDEO_DIR = "videos_enduro_final"
NB_VIDEO_EPISODES = 1


WEIGHTS_FILE = "dqn_Enduro-v0_double_double_continuacion_5_3_weights.h5f"
VARIANTE = "double"

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4
MEMORY_LIMIT = 300000
NB_STEPS_WARMUP = 10000
NB_STEPS_ANNEAL = 40000

TARGET_UPDATE = 10000
TRAIN_INTERVAL = 4
LEARNING_RATE = 0.00025
GAMMA = 0.99
SEED = 123

VARIANT_CONFIGS = {
    "base": dict(double_dqn=False, dueling=False),
    "double": dict(double_dqn=True, dueling=False),
    "dueling": dict(double_dqn=False, dueling=True),
}


class AtariProcessor(Processor):
    def process_observation(self, observation):
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert("L")
        return np.array(img).astype("uint8")

    def process_state_batch(self, batch):
        return batch.astype("float32") / 255.0

    def process_reward(self, reward):
        return np.clip(reward, -1.0, 1.0)


def build_model(nb_actions):
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE

    frames = Input(shape=input_shape, name="frames")
    x = Permute((2, 3, 1), name="channels_last")(frames)

    x = Convolution2D(32, (8, 8), strides=(4, 4), activation="relu", name="conv1")(x)
    x = Convolution2D(64, (4, 4), strides=(2, 2), activation="relu", name="conv2")(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation="relu", name="conv3")(x)

    x = Flatten(name="flatten")(x)
    x = Dense(512, activation="relu", name="dense512")(x)
    q_values = Dense(nb_actions, activation="linear", name="q_values")(x)

    return Model(inputs=frames, outputs=q_values, name="dqn_enduro")


def build_agent(nb_actions, double_dqn=False, dueling=False, dueling_type="avg"):
    model = build_model(nb_actions)

    memory = SequentialMemory(
        limit=MEMORY_LIMIT,
        window_length=WINDOW_LENGTH
    )

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.1,
        value_test=0.01,
        nb_steps=NB_STEPS_ANNEAL,
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=memory,
        processor=AtariProcessor(),
        nb_steps_warmup=NB_STEPS_WARMUP,
        gamma=GAMMA,
        target_model_update=TARGET_UPDATE,
        train_interval=TRAIN_INTERVAL,
        delta_clip=1.0,
        enable_double_dqn=double_dqn,
        enable_dueling_network=dueling,
        dueling_type=dueling_type,
    )

    try:
        optimizer = Adam(learning_rate=LEARNING_RATE)
    except TypeError:
        optimizer = Adam(lr=LEARNING_RATE)

    dqn.compile(optimizer, metrics=["mae"])
    return dqn


def main():
    os.chdir(Path(PROJECT_ROOT).expanduser().resolve())

    if not (
        os.path.exists(WEIGHTS_FILE)
        or os.path.exists(WEIGHTS_FILE + ".index")
    ):
        raise FileNotFoundError(f"No encuentro los pesos: {WEIGHTS_FILE}")

    print("Cargando pesos:", WEIGHTS_FILE)

    video_dir = Path(VIDEO_DIR)
    video_dir.mkdir(parents=True, exist_ok=True)

    env = wrappers.Monitor(
        gym.make(ENV_NAME),
        str(video_dir),
        force=True,
        video_callable=lambda episode_id: True,
    )

    try:
        env.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        nb_actions = env.action_space.n

        K.clear_session()

        dqn = build_agent(
            nb_actions,
            **VARIANT_CONFIGS[VARIANTE],
        )

        dqn.load_weights(WEIGHTS_FILE)

        print("Grabando vídeo...")
        dqn.test(
            env,
            nb_episodes=NB_VIDEO_EPISODES,
            visualize=False,
            verbose=1,
        )

        print("Vídeo guardado en:", video_dir.resolve())

    finally:
        env.close()


if __name__ == "__main__":
    main()