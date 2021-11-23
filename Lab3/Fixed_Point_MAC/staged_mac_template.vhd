-------------------------------------------------------------------------
-- Matthew Dwyer
-- Department of Electrical and Computer Engineering
-- Iowa State University
-------------------------------------------------------------------------


-- staged_mac.vhd
-------------------------------------------------------------------------
-- DESCRIPTION: This file contains a basic staged axi-stream mac unit. It
-- multiplies two integer/Q values togeather and accumulates them.
--
-- NOTES:
-- 10/25/21 by MPD::Inital template creation
-------------------------------------------------------------------------

library work;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use std.textio.all;

entity staged_mac is
  generic(
      -- Parameters of mac
      C_DATA_WIDTH : integer := 32
    );
  port( ARESETN	        : in	std_logic; --reset    
        ACLK            : in    std_logic; -- Main CLK
        -- AXIS slave data interface
		SD_AXIS_TREADY	: out	std_logic; --Send signal to master when slave is ready to receive data
		SD_AXIS_TDATA	: in	std_logic_vector(C_DATA_WIDTH*2-1 downto 0);  -- Packed data input from master
		SD_AXIS_TLAST	: in	std_logic; --Last data frame in valid data that informs slave to complete current operation and prepare for a new data stream
        SD_AXIS_TUSER   : in    std_logic;  -- Should we treat this first value in the stream as an inital accumulate value? OPTIONAL
		SD_AXIS_TVALID	: in	std_logic; --when high SD_AXIS_TDATA is valid 
        SD_AXIS_TID     : in    std_logic_vector(7 downto 0); --Identifier for sent data OPTIONAL

        -- AXIS master accumulate result out interface
		MO_AXIS_TVALID	: out	std_logic; --high when MO_AXIS_DATA is valid
		MO_AXIS_TDATA	: out	std_logic_vector(C_DATA_WIDTH-1 downto 0); --data to send to master
		MO_AXIS_TLAST	: out	std_logic; --Last bit to inform master of next operation
		MO_AXIS_TREADY	: in	std_logic; --when high, master is ready to receive data
		MO_AXIS_TID     : out   std_logic_vector(7 downto 0) --OPTIONAL
    );


end staged_mac;

   

architecture mixed of staged_mac is
    -- Internal Signals
	
	
	-- Mac state
    type STATE_TYPE is (WAIT_FOR_VALUES, PROCESSING_VALUES, WAITING_TO_SEND_VALUES, SENDING_DATA);
    signal state : STATE_TYPE;
	
	-- Debug signals, make sure we aren't going crazy
    signal mac_debug : std_logic_vector(31 downto 0) := x"00000000";


    -- N-bit register component declaration
    component dffg is
        generic(
            -- Parameters of mac
            C_DATA_WIDTH : integer := 32
          );
          port(i_CLK        : in    std_logic;     -- Clock input
               i_RST        : in    std_logic;     -- Reset input
               i_WE         : in    std_logic;     -- Write enable input
               i_D          : in    std_logic_vector(C_DATA_WIDTH-1 downto 0);   -- Data value input
               o_Q          : out	std_logic_vector(C_DATA_WIDTH-1 downto 0)    -- Data value output
              ); 
    end component; 

    -- Interface signals


    -- Internal signals
    
    --accumulate signal from reg
    signal accumulate : std_logic_vector(C_DATA_WIDTH-1 downto 0) := x"00000000";
    --inputs A and B will be the inputs to the MAC
	signal A : std_logic_vector(C_DATA_WIDTH-1 downto 0);
    signal B : std_logic_vector(C_DATA_WIDTH-1 downto 0);
    --Signal that holds A*B
    signal MULT : std_logic_vector(2*(C_DATA_WIDTH)-1 downto 0);
    --Signal to reset register.
    signal reset_reg : std_logic := '0';
    signal mult_add : std_logic_vector((C_DATA_WIDTH-1) downto 0) := x"00000000";

begin

    --Component for register.
    register_acc: dffg
        port MAP(i_CLK  => ACLK,
                 i_RST  => reset_reg,
                 i_WE   => '1',
                 i_D    => mult_add,
                 o_Q    => accumulate
                );

	
	-- Debug Signals
    --mac_debug <= x"00000000";  -- Double checking sanity    
    
   process (ACLK, MULT, mult_add, accumulate, A, B, reset_reg) is
   begin 
    if rising_edge(ACLK) then  -- Rising Edge

      -- Reset values if reset is low
      if ARESETN = '0' then  -- Reset
        state <= WAIT_FOR_VALUES;

      else
        case state is  -- State
            -- Wait here until we receive values
            when WAIT_FOR_VALUES =>
                -- Wait here until we recieve valid values
                SD_AXIS_TREADY <= '1';
                --report "Here";
                --state <= PROCESSING_VALUES;
                if(SD_AXIS_TVALID) then
                    report "Here";
                    -- Break input into data A and B for MAC
                    A <= SD_AXIS_TDATA(C_DATA_WIDTH-1 downto 0);
                    B <= SD_AXIS_TDATA((C_DATA_WIDTH*2)-1 downto C_DATA_WIDTH);
                    state <= PROCESSING_VALUES;
                elsif(not SD_AXIS_TVALID) then
                    report "NOT Here";
                    --state <= WAIT_FOR_VALUES;
                end if;

			-- Other stages go here	
            when PROCESSING_VALUES =>
                --MAC here

                report "In Processing Values State.";
                reset_reg <= '0';
                if(not SD_AXIS_TVALID) then
                    state <= WAIT_FOR_VALUES;
                end if;

                --Multiply and accumulate 
                MULT <= std_logic_vector(signed(A) * signed(B));
                mult_add <= std_logic_vector(signed(MULT(C_DATA_WIDTH-1 downto 0)) + signed(accumulate)); --Take first 32 bits of MULT to truncate and prevent overflow
                
                if(SD_AXIS_TLAST and MO_AXIS_TREADY) then
                    SD_AXIS_TREADY <= '0';
                    state <= SENDING_DATA;
                elsif (SD_AXIS_TLAST and not MO_AXIS_TREADY) then
                    SD_AXIS_TREADY <= '0';
                    state <= WAITING_TO_SEND_VALUES;
                elsif (not SD_AXIS_TLAST) then
                    state <= PROCESSING_VALUES;
                end if;

            when WAITING_TO_SEND_VALUES =>
        
                if(MO_AXIS_TREADY) then
                    state <= SENDING_DATA;
                elsif (not MO_AXIS_TREADY) then
                    state <= WAITING_TO_SEND_VALUES;
                end if;

            when SENDING_DATA =>
                SD_AXIS_TREADY <= '0';
                MO_AXIS_TDATA <= accumulate;
                MO_AXIS_TVALID <= '1';
                if(not MO_AXIS_TREADY) then
                    state <= WAITING_TO_SEND_VALUES;
                end if;
                reset_reg <= '1';
                if(not SD_AXIS_TVALID) then
                    state <= WAIT_FOR_VALUES;
                elsif (SD_AXIS_TREADY and SD_AXIS_TVALID) then
                    state <= PROCESSING_VALUES;
                end if;

        
            when others =>
                state <= WAIT_FOR_VALUES;
                -- Not really important, this case should never happen
                -- Needed for proper synthisis         
        end case;  -- State
      end if;  -- Reset

    end if;  -- Rising Edge
   end process;
end architecture mixed;
