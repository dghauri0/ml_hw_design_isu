-------------------------------------------------------------------------
-- Matthew Dwyer
-- Department of Electrical and Computer Engineering
-- Iowa State University
-------------------------------------------------------------------------


-- piped_mac.vhd
-------------------------------------------------------------------------
-- DESCRIPTION: This file contains a basic piplined axi-stream mac unit. It
-- multiplies two integer/Q values togeather and accumulates them.
--
-- NOTES:
-- 10/25/21 by MPD::Inital template creation
-------------------------------------------------------------------------

library work;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity piped_mac is
  generic(
      -- Parameters of mac
      C_DATA_WIDTH : integer := 32
    );
	port (
        ACLK	: in	std_logic;
		ARESETN	: in	std_logic;       

        -- AXIS slave data interface
		SD_AXIS_TREADY	: out	std_logic;
		SD_AXIS_TDATA	: in	std_logic_vector(C_DATA_WIDTH*2-1 downto 0);  -- Packed data input
		SD_AXIS_TLAST	: in	std_logic;
        SD_AXIS_TUSER   : in    std_logic;  -- Should we treat this first value in the stream as an inital accumulate value?
		SD_AXIS_TVALID	: in	std_logic;
        SD_AXIS_TID     : in    std_logic_vector(7 downto 0);

        -- AXIS master accumulate result out interface
		MO_AXIS_TVALID	: out	std_logic;
		MO_AXIS_TDATA	: out	std_logic_vector(C_DATA_WIDTH-1 downto 0);
		MO_AXIS_TLAST	: out	std_logic;
		MO_AXIS_TREADY	: in	std_logic;
		MO_AXIS_TID     : out   std_logic_vector(7 downto 0)
    );

attribute SIGIS : string; 
attribute SIGIS of ACLK : signal is "Clk"; 

end piped_mac;


architecture behavioral of piped_mac is
    -- Internal Signals
	signal MULT       : std_logic_vector(2*(C_DATA_WIDTH)-1 downto 0) := x"0000000000000000";
    signal accumulate : std_logic_vector(C_DATA_WIDTH-1 downto 0) := x"00000000"; 

    --input piped signals
    signal sd_axis_tlast_s2  : std_logic := '0';
    signal sd_axis_tvalid_s2 : std_logic := '0';
    signal mo_axis_tready_s2 : std_logic := '0';
    
	-- Mac stages
    type PIPE_STAGES is (MULT_STAGE, ACC_STAGE);

	
	-- Debug signals, make sure we aren't going crazy
    signal mac_debug : std_logic_vector(31 downto 0) := x"00000000"; 

begin

    -- Interface signals


    -- Internal signals
	
	

   
   process (ACLK) is
   begin 
    if rising_edge(ACLK) then  -- Rising Edge

      -- Reset values if reset is low
      if ARESETN = '0' then  -- Reset
		
      else
        for i in PIPE_STAGES'left to PIPE_STAGES'right loop
            case i is  -- Stages
                when MULT_STAGE =>
					-- Pipline stage 0 - Multiply Stage

                    -- Passing "control" signals to accumulate stage (register infer)
                    sd_axis_tlast_s2  <= SD_AXIS_TLAST;
                    sd_axis_tvalid_s2 <= SD_AXIS_TVALID;
                    mo_axis_tready_s2 <= MO_AXIS_TREADY;
                    
                    if (SD_AXIS_TVALID = '1') then 
                        MULT <= std_logic_vector(unsigned(SD_AXIS_TDATA(2*(C_DATA_WIDTH)-1 downto C_DATA_WIDTH)) * unsigned(SD_AXIS_TDATA(C_DATA_WIDTH-1 downto 0)));
                    end if;
                    
                when ACC_STAGE =>
                    -- Pipline stage 1 - Accumulate
                    if(not mo_axis_tready_s2 = '1') then
                        SD_AXIS_TREADY <= '0';
                    else
                        SD_AXIS_TREADY <= '1';
                    end if;

                    if ((sd_axis_tvalid_s2 = '1') and not (sd_axis_tlast_s2 = '1')) then
                        MO_AXIS_TLAST  <= '0';
                        MO_AXIS_TVALID <= '0';
			            mac_debug <= MULT(C_DATA_WIDTH-1 downto 0);
                        accumulate     <= std_logic_vector(unsigned(MULT(C_DATA_WIDTH-1 downto 0)) + unsigned(accumulate));
                    elsif ((sd_axis_tlast_s2 = '1') and (sd_axis_tvalid_s2 = '1')) then
                        MO_AXIS_TLAST  <= '1';
                        MO_AXIS_TVALID <= '1';
                        MO_AXIS_TDATA  <= std_logic_vector(unsigned(MULT(C_DATA_WIDTH-1 downto 0)) + unsigned(accumulate));
                        accumulate     <= (others => '0');
                    end if;

                    end case;  -- Stages
		end loop;  -- Stages
      end if;  -- Reset

    end if;  -- Rising Edge
   end process;
end architecture behavioral;
