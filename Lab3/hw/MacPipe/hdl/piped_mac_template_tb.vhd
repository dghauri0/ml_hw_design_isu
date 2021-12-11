-------------------------------------------------------------------------
-- Henry Duwe
-- Department of Electrical and Computer Engineering
-- Iowa State University
-------------------------------------------------------------------------
-- piped_mac_template_db.vhd
-------------------------------------------------------------------------
-- DESCRIPTION: This file contains a testbench for the TPU MAC unit.
--              
-- 01/03/2020 by H3::Design created.
-------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_textio.all;  -- For logic types I/O
library std;
use std.env.all;                -- For hierarchical/external signals
use std.textio.all;             -- For basic I/O

-- Usually name your testbench similar to below for clarity tb_<name>
-- TODO: change all instances of tb_TPU_MV_Element to reflect the new testbench.
entity piped_mac_tb is
  generic(gCLK_HPER   : time := 10 ns; C_DATA_WIDTH : integer := 32);   -- Generic for half of the clock cycle period
end piped_mac_tb;

architecture mixed of piped_mac_tb is

-- Define the total clock period time
constant cCLK_PER  : time := gCLK_HPER * 2;

-- We will be instantiating our design under test (DUT), so we need to specify its
-- component interface.
-- TODO: change component declaration as needed.
component piped_mac is
  generic(
      -- Parameters of mac
      C_DATA_WIDTH : integer := 32
    );
  port( ARESETN	        : in	std_logic;       
        ACLK            : in  std_logic; -- Main CLK
        -- AXIS slave data interface
		    SD_AXIS_TREADY	: out	std_logic; --Send signal to master when slave is ready to receive data
		    SD_AXIS_TDATA	  : in	std_logic_vector(C_DATA_WIDTH*2-1 downto 0);  -- Packed data input from master
		    SD_AXIS_TLAST	  : in	std_logic; --Last data frame in valid data that informs slave to complete current operation and prepare for a new data stream
        SD_AXIS_TUSER   : in  std_logic;  -- Should we treat this first value in the stream as an inital accumulate value? OPTIONAL
		    SD_AXIS_TVALID	: in	std_logic; --when high SD_AXIS_TDATA is valid 
        SD_AXIS_TID     : in  std_logic_vector(7 downto 0); --Identifier for sent data OPTIONAL

        -- AXIS master accumulate result out interface
		    MO_AXIS_TVALID	: out	std_logic; --high when MO_AXIS_DATA is valid
		    MO_AXIS_TDATA	  : out	std_logic_vector(C_DATA_WIDTH-1 downto 0); --data to send to master
		    MO_AXIS_TLAST	  : out	std_logic; --Last bit to inform master of next operation
		    MO_AXIS_TREADY	: in	std_logic; --when high, master is ready to receive data
		    MO_AXIS_TID     : out std_logic_vector(7 downto 0) --OPTIONAL
    );


end component;

-- Create signals for all of the inputs and outputs of the file that you are testing
-- := '0' or := (others => '0') just make all the signals start at an initial value of zero
signal ACLK, reset : std_logic := '0';

-- TODO: change input and output signals as needed.
signal s_ARESETN          : std_logic := '0';

-- Slave Testbench Signals
signal s_SD_AXIS_TREADY   : std_logic := '0';
signal s_SD_AXIS_TDATA    : std_logic_vector(C_DATA_WIDTH*2-1 downto 0) := x"0000000000000000";
signal s_SD_AXIS_TLAST    : std_logic := '0';
signal s_SD_AXIS_TUSER    : std_logic := '0';
signal s_SD_AXIS_TVALID   : std_logic := '0';
signal s_SD_AXIS_TID      : std_logic_vector(7 downto 0) := x"00";

-- Master Testbench Signals
signal s_MO_AXIS_TVALID	: std_logic := '0';
signal s_MO_AXIS_TDATA	:	std_logic_vector(C_DATA_WIDTH-1 downto 0) := x"00000000";
signal s_MO_AXIS_TLAST	:	std_logic := '0';
signal s_MO_AXIS_TREADY	:	std_logic := '0';
signal s_MO_AXIS_TID    : std_logic_vector(7 downto 0);



  -- TODO: Actually instantiate the component to test and wire all signals to the corresponding
  -- input or output. Note that DUT0 is just the name of the instance that can be seen 
  -- during simulation. What follows DUT0 is the entity name that will be used to find
  -- the appropriate library component during simulation loading.
begin
  
  DUT0: piped_mac
  port map(
    ACLK    => ACLK,
		ARESETN	=> reset,

    -- AXIS slave data interface
		SD_AXIS_TREADY  => s_SD_AXIS_TREADY,	
		SD_AXIS_TDATA	  => s_SD_AXIS_TDATA,
		SD_AXIS_TLAST	  => s_SD_AXIS_TLAST,
    SD_AXIS_TUSER   => s_SD_AXIS_TUSER,
		SD_AXIS_TVALID	=> s_SD_AXIS_TVALID,
    SD_AXIS_TID     => s_SD_AXIS_TID,

    -- AXIS master accumulate result out interface
		MO_AXIS_TVALID	=> s_MO_AXIS_TVALID,
		MO_AXIS_TDATA	  => s_MO_AXIS_TDATA,
		MO_AXIS_TLAST	  => s_MO_AXIS_TLAST,
		MO_AXIS_TREADY  => s_MO_AXIS_TREADY,
		MO_AXIS_TID     => s_MO_AXIS_TID);     
  --You can also do the above port map in one line using the below format: http://www.ics.uci.edu/~jmoorkan/vhdlref/compinst.html

  
  --This first process is to setup the clock for the test bench
  P_CLK: process
  begin
    ACLK <= '1';         -- clock starts at 1
    wait for gCLK_HPER; -- after half a cycle
    ACLK <= '0';         -- clock becomes a 0 (negative edge)
    wait for gCLK_HPER; -- after half a cycle, process begins evaluation again
  end process;

  -- This process resets the sequential components of the design.
  -- It is held to be 1 across both the negative and positive edges of the clock
  -- so it works regardless of whether the design uses synchronous (pos or neg edge)
  -- or asynchronous resets.
  P_RST: process
  begin
  	reset <= '1';   
    wait for gCLK_HPER/2;
	  reset <= '0';
    wait for gCLK_HPER;
	reset <= '1';
	wait;
  end process;  
  
  P_TEST_CASES: process
  begin
    wait for gCLK_HPER/2; -- for waveform clarity, I prefer not to change inputs on clk edges

    -- Test case 1:
    --Wait for reset to finsih
    wait for gCLK_HPER/2; --5 ns
    wait for gCLK_HPER;   --10 ns

    --Begin multiplying and accumulating
    --Test case 1
    --Expect: 84
    s_MO_AXIS_TREADY <= '1';
    --s_SD_AXIS_TREADY <= '1';
    s_SD_AXIS_TVALID <= '1';
    s_SD_AXIS_TDATA  <= x"0000000800000008";
    wait for gCLK_HPER*2;
    
    s_SD_AXIS_TDATA  <= x"0000000400000004";
    wait for gCLK_HPER*2;
    
    s_SD_AXIS_TDATA  <= x"0000000200000002";
    s_SD_AXIS_TLAST  <= '1';    
    wait for gCLK_HPER*2;

    s_SD_AXIS_TLAST  <= '0';
    s_SD_AXIS_TDATA  <= x"0000000000000000";
    wait for gCLK_HPER*2;

    --test case 2
    --stall for 1 cycles since incoming data isn't valid
    s_SD_AXIS_TVALID <= '0';
    s_SD_AXIS_TDATA  <= x"FFFFFFFFFFFFFFFF";
    s_MO_AXIS_TREADY <= '1';
    wait for gCLK_HPER;
    s_SD_AXIS_TVALID <= '1';
    wait for gCLK_HPER;
    --we should now be receiving data set in last cycle since it hasn't changed. This data will cause overflow and be conactenated.
    wait for gCLK_HPER*2;
    
    s_SD_AXIS_TDATA <= x"0000000400000004";
    wait for gCLK_HPER*2;

    s_SD_AXIS_TDATA <= x"0000000200000002";
    s_SD_AXIS_TLAST <= '1';
    wait for gCLK_HPER*2;

    
  end process;

end mixed;
