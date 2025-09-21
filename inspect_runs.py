from tensorboard.backend.event_processing import event_accumulator
import os



ea = event_accumulator.EventAccumulator('runs/data13/1_sim_2000_200_20_0.45_0.45_0.7_1/vae_data13/1_sim_2000_200_20_0.45_0.45_0.7_1_20240605-142303/events.out.tfevents.1717728.DESKTOP-UB5H6D4.17060.0')

ea.Reload()  # loads events from file

scalars = ea.scalars
print(scalars)

if __name__ == "__main__":
    pass
