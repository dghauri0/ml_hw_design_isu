-------------------------------------------------------------------------
-- Matthew Dwyer
-- Department of Electrical and Computer Engineering
-- Iowa State University
-------------------------------------------------------------------------


-- 4x4_Acc.vhd
-------------------------------------------------------------------------
-- DESCRIPTION: This file contains a 4x4 matrix-matrix accelerator supporting
-- addition and multiplication with support for an initial accumulation matrix.
--
-- NOTES:
-- 11/17/21 by MPD::Inital template creation
-------------------------------------------------------------------------

library work;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity Acc_4x4 is
  generic(
      -- Parameters of acc
      C_DATA_WIDTH : integer := 32
    );
	port (
        ACLK	: in	std_logic;
		ARESETN	: in	std_logic
    );

attribute SIGIS : string; 
attribute SIGIS of ACLK : signal is "Clk"; 

end Acc_4x4;


architecture behavioral of Acc_4x4 is
    -- Internal Signals
	
	
	-- Acc state
    type STATE_TYPE is (WAIT_FOR_VALUES);
    signal state : STATE_TYPE;
	
	-- Debug signals, make sure we aren't going crazy
    signal acc_debug : std_logic_vector(31 downto 0);

begin

    -- Interface signals


    -- Internal signals
	
	
	-- Debug Signals
    acc_debug <= x"00000000";  -- Double checking sanity
   
   process (ACLK) is
   begin 
    if rising_edge(ACLK) then  -- Rising Edge

      -- Reset values if reset is low
      if ARESETN = '0' then  -- Reset
        state       <= WAIT_FOR_VALUES;

      else
        case state is  -- State
            -- Wait here until we receive values
            when WAIT_FOR_VALUES =>
                -- Wait here until we recieve valid values
			
			
			-- Other stages go here	
			
            when others =>
                state <= WAIT_FOR_VALUES;
                -- Not really important, this case should never happen
                -- Needed for proper synthisis         
        end case;  -- State
      end if;  -- Reset

    end if;  -- Rising Edge
   end process;
end architecture behavioral;
