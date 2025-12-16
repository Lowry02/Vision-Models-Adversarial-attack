# Adversarial Attacks Against LVLMs: A Case Study on Qwen/Qwen2.5-VL-3B-Instruct

The projecty involves the implementation and evaluation of adversarial attacks targeting the `Qwen/Qwen2.5-VL-3B-Instruct` Large Vision-Language Model (LVLM). The goal of these attacks is to "jailbreak" the model, causing it to bypass its safety filters and respond to a harmful instruction by maximizing the probability of the token "Sure" being generated.

The project explores two types of adversarial attacks:
1.  **Image Adversarial Attack**: An adversarial noise pattern is generated and applied to an image.
2.  **Text Adversarial Attack**: An adversarial suffix is generated and appended to the text prompt.

## Summary of Results

The adversarial suffix approach consistently outperformed the adversarial image method in maximizing the target token generation. This suggests that textual manipulations are more effective than image perturbations for inducing undesirable behavior in `Qwen/Qwen2.5-VL-3B-Instruct` under the given constraints.

## Algorithm Idea: Genetic Algorithm (GA)

Both attacks are implemented using a Genetic Algorithm to find an effective adversarial input. The algorithm evolves a population of solutions over several generations, guided by the following project-specific setup:

-   **Population**: 
    -   For the **Image Attack**, the population consists of tensors of random noise which are added to a blank image.
    -   For the **Text Attack**, the population is a set of sequences of random token IDs, which are used as a suffix to the text prompt.

-   **Fitness Function**: The fitness of each individual is calculated as the probability assigned by the model to the token "Sure" as the first token of its response. A higher probability indicates a more successful attack.

-   **Selection**: **Tournament Selection** is used to choose the individuals for creating the next generation.

-   **Crossover**: **One-Point Crossover** is applied to the selected parents to create offspring.

-   **Mutation**: Small, random changes are introduced into the offspring to maintain diversity.

-   **Elitism**: A portion of the best-performing individuals are carried over to the next generation to preserve the most successful solutions.

## Evaluation

The effectiveness of the attacks was measured using the following metrics:

-   **Fitness**: The likelihood of generating the "Sure" token.
-   **"Sure" Count**: The number of times the token "Sure" appeared in the model's output during each iteration.

## Files

-   `run.py`: The main script that orchestrates the adversarial attacks.
-   `QwenChat.py`: A wrapper class for the `Qwen/Qwen2.5-VL-3B-Instruct` model.
-   `ImageAdversarialAttack.py`: Implements the GA for the image-based attack.
-   `TextAdversarialAttack.py`: Implements the GA for the text-based attack.
-   `Cusin Lorenzo - Report.pdf`: A detailed report on the project, methodology, and results.
-   

---

More details in the report.