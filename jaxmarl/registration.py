from .environments import (
    SimpleMPE,
    SimpleTagMPE,
    SimpleWorldCommMPE,
    SimpleSpreadMPE,
    SimpleCryptoMPE,
    SimpleSpeakerListenerMPE,
    SimpleFacmacMPE,
    SimpleFacmacMPE3a,
    SimpleFacmacMPE6a,
    SimpleFacmacMPE9a,
    SimplePushMPE,
    SimpleAdversaryMPE,
    SimpleReferenceMPE,
    AugmentedMPE,
    Hanabi,

)



def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered jaxmarl environments.")

    # 1. MPE PettingZoo Environments
    if env_id == "MPE_simple_v3":
        env = SimpleMPE(**env_kwargs)
    elif env_id == "MPE_simple_tag_v3":
        env = SimpleTagMPE(**env_kwargs)
    elif env_id == "MPE_simple_world_comm_v3":
        env = SimpleWorldCommMPE(**env_kwargs)
    elif env_id == "MPE_simple_spread_v3":
        env = SimpleSpreadMPE(**env_kwargs)
    elif env_id == "MPE_simple_crypto_v3":
        env = SimpleCryptoMPE(**env_kwargs)
    elif env_id == "MPE_simple_speaker_listener_v4":
        env = SimpleSpeakerListenerMPE(**env_kwargs)
    elif env_id == "MPE_simple_push_v3":
        env = SimplePushMPE(**env_kwargs)
    elif env_id == "MPE_simple_adversary_v3":
        env = SimpleAdversaryMPE(**env_kwargs)
    elif env_id == "MPE_simple_reference_v3":
        env = SimpleReferenceMPE(**env_kwargs)
    elif env_id == "MPE_simple_facmac_v1":
        env = SimpleFacmacMPE(**env_kwargs)
    elif env_id == "MPE_simple_facmac_3a_v1":
        env = SimpleFacmacMPE3a(**env_kwargs)
    elif env_id == "MPE_simple_facmac_6a_v1":
        env = SimpleFacmacMPE6a(**env_kwargs)
    elif env_id == "MPE_simple_facmac_9a_v1":
        env = SimpleFacmacMPE9a(**env_kwargs)
    elif env_id == "AugmentedMPE":
        env = AugmentedMPE(**env_kwargs)

    # 6. Hanabi
    elif env_id == "hanabi":
        env = Hanabi(**env_kwargs)


    return env

registered_envs = [
    "MPE_simple_v3",
    "MPE_simple_tag_v3",
    "MPE_simple_world_comm_v3",
    "MPE_simple_spread_v3",
    "MPE_simple_crypto_v3",
    "MPE_simple_speaker_listener_v4",
    "MPE_simple_push_v3",
    "MPE_simple_adversary_v3",
    "MPE_simple_reference_v3",
    "MPE_simple_facmac_v1",
    "MPE_simple_facmac_3a_v1",
    "MPE_simple_facmac_6a_v1",
    "MPE_simple_facmac_9a_v1",
    "AugmentedMPE",

    "hanabi",
]
