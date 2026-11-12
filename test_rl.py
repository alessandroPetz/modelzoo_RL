from seanergys_modelzoo.models.tasks.power_consumption_manage.RL.rl_agent_sim3_hard_constraint import (
    RLAgentSim3HardConstraint,
)

USE_DUMMY = False   # metti False per usare il dataset normale
SAVE_PATH = "saved_models/rl_agent_sim3_hard_constraint"


if __name__ == "__main__":
    # Train config
    # model_obj, train_dataloader = RLAgentSim3HardConstraint.get_train_config(
    #     "pm100",
    #     get_dummy_data=USE_DUMMY,
    # )


    model_obj, train_dataloader = (
        RLAgentSim3HardConstraint.get_train_config(
            "pm100",
            get_dummy_data=USE_DUMMY,

            # Solo un giorno di workload
            window_start="2020-05-25T00:00:00Z",
            window_end="2020-05-26T00:00:00Z",
            episode_hours=24,
            stride_hours=24,

            # Training breve per verificare l'integrazione
            max_episodes=1,
            total_timesteps=5000,
            n_steps=512,
            batch_size=128,
        )
    )


    # Train
    train_results = model_obj.train(train_dataloader)
    print("=== TRAIN RESULTS ===")
    print(train_results)

    # Save
    model_obj.save(SAVE_PATH)
    print(f"Model saved to: {SAVE_PATH}")

    # Test config
    test_dataloader = RLAgentSim3HardConstraint.get_test_config(
        model=model_obj,
        dataset_name="pm100",
        get_dummy_data=USE_DUMMY,
    )

    # Evaluate
    eval_results = model_obj.evaluate(test_dataloader)
    print("=== EVAL RESULTS ===")
    print(eval_results)
