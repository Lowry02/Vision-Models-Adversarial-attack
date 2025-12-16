from QwenChat import QwenChat
from ImageAdversarialAttack import ImageAdversarialAttack
from TextAdversarialAttack import TextAdversarialAttack
import torch
import csv

chat = QwenChat()

fieldnames = ['run', 'iteration', 'best_fitness', 'sure_count']
with open("image_adv_info.csv", 'w', newline='') as csvfile:
  logger = csv.DictWriter(csvfile, fieldnames=fieldnames)
  logger.writeheader()
  
  adversarial_attack = ImageAdversarialAttack(
    chat,
    image_size=(300,300),
    block_side=50,
    logger=logger
  )
  
  for i in range(10):
    print(f"-- RUN {i} --")
    population = adversarial_attack.run(
      run=i,
      population_size=30,
      mutation_rate=0.08,
      tournament_pressure=3,
      elitism_rate=0.08,
      n_iterations=50,
    )

    torch.save(population, f"adv_images_{i}.pt")
    

fieldnames = ['run', 'iteration', 'best_fitness', 'sure_count']
with open("text_adv_info.csv", 'w', newline='') as csvfile:
  logger = csv.DictWriter(csvfile, fieldnames=fieldnames)
  logger.writeheader()
  
  adversarial_attack = TextAdversarialAttack(chat, logger=logger)

  for i in range(10):
    print(f"-- RUN {i} --")
    population = adversarial_attack.run(
      run=i,
      population_size=30,
      adv_prefix_length=25,
      mutation_rate=0.08,
      tournament_pressure=5,
      elitism_rate=0.08,
      n_iterations=50,
    )
    torch.save(population, f"adv_suffix_{i}.pt")