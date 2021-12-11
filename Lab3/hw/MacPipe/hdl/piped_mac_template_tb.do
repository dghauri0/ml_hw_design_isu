# Start sim
vsim -voptargs=+acc work.piped_mac_tb

# Add all signals
add wave -noupdate -radix unsigned /piped_mac_tb/DUT0/*

# Run for x timesteps (default is 1ns per timestep, but this can be modified so be aware).
run 100 ns 