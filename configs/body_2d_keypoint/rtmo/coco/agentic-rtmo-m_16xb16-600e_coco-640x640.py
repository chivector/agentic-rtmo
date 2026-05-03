_base_ = ['./rtmo-m_16xb16-600e_coco-640x640.py']

# Keep baseline training recipe and only add lightweight agentic loop.
model = dict(
    head=dict(
        type='RTMOHead',
        dcc_cfg=dict(
            agentic_cfg=dict(
                enabled=True,
                num_iters=2,
                critic_hidden_dim=64,
                actor_hidden_dim=128,
                residual_scale=0.5,
            ))))
