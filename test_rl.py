from seanergys_modelzoo.models.tasks.power_consumption_manage.RL.rl_agent_sim3_hard_constraint import (
    RLAgentSim3HardConstraint,
)

USE_DUMMY = True   # metti False per usare il dataset normale
SAVE_PATH = "saved_models/rl_agent_sim3_hard_constraint"


if __name__ == "__main__":
    # Train config
    model_obj, train_dataloader = RLAgentSim3HardConstraint.get_train_config(
        "pm100",
        get_dummy_data=USE_DUMMY,
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
