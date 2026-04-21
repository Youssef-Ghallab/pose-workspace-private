dataset_info = dict(
    dataset_name='soccernet_fhd',
    paper_info=dict(
        author='SoccerNet',
        title='SoccerNet SpiideoSynLoc Full HD Pose',
        container='Custom dataset',
        year='2024',
        homepage='https://www.soccer-net.org/',
    ),
    keypoint_info={
        0: dict(
            name='body_anchor',
            id=0,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        1: dict(
            name='ground_contact',
            id=1,
            color=[255, 128, 0],
            type='lower',
            swap=''),
    },
    skeleton_info={
        0: dict(
            link=('body_anchor', 'ground_contact'),
            id=0,
            color=[51, 153, 255]),
    },
    joint_weights=[1.0, 1.0],
    sigmas=[0.05, 0.05],
)
