function [ r ] = MATLABStandaloneApplication( args )

if ~exist('args', 'var')
    args = [];
end

% Initialize the OpticStudio connection
TheApplication = InitConnection();
if isempty(TheApplication)
    % failed to initialize a connection
    r = [];
else
    try
        r = BeginApplication(TheApplication, args);
        CleanupConnection(TheApplication);
    catch err
        CleanupConnection(TheApplication);
        rethrow(err);
    end
end
end


function [r] = BeginApplication(TheApplication, args)

import ZOSAPI.*;

TheSystem = TheApplication.PrimarySystem;

% Add your custom code here...

%filename
testFile = 'C:\Users\Matthew.Leung\Documents\Research\Shaojie\ZOS-API_test\DMD_MOS_20190813(for Geometric Imaging Analysis).zmx';
TheSystem.LoadFile(testFile,false);

for i = 125:50:500
    % open new geometric image analysis
    TheAnalyses = TheSystem.Analyses; %create new instance of the analyses class
    GeoIma = TheAnalyses.New_GeometricImageAnalysis();

    % get settings
    GeoImaSettings = GeoIma.GetSettings();

    GeoImaSettings.FieldSize = 14;
    GeoImaSettings.RaysX1000 = 10;
    GeoImaSettings.ShowAs = ZOSAPI.Analysis.GiaShowAsTypes.SpotDiagram;
    GeoImaSettings.Field.SetFieldNumber(1);
    GeoImaSettings.Wavelength.UseAllWavelengths();
    %GeoImaSettings.Surface.SetSurfaceNumber(54);
    GeoImaSettings.Surface.UseImageSurface();

    % change IMA file
    GeoImaSettings.File = append('dmd_', int2str(i), '_MOS.IMA');

    % apply the new settings
    GeoIma.ApplyAndWaitForCompletion(); %be sure to have this line

    % get results
    GeoImaResults = GeoIma.GetResults();

    % save results to text file in Zemax\Samples\API\Matlab
    %GeoImaResults.GetTextFile(System.String.Concat(TheApplication.SamplesDir, '\test.txt'));
    filename = append('\data_', int2str(i), '.txt');
    directory = 'C:\Users\Matthew\Desktop\Research\Fake_Data\new_fit\data';
    GeoImaResults.GetTextFile(System.String.Concat(directory, filename));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % open new geometric image analysis
    TheAnalyses = TheSystem.Analyses; %create new instance of the analyses class
    GeoIma = TheAnalyses.New_GeometricImageAnalysis();

    % get settings
    GeoImaSettings = GeoIma.GetSettings();

    GeoImaSettings.FieldSize = 14;
    GeoImaSettings.RaysX1000 = 10;
    GeoImaSettings.ShowAs = ZOSAPI.Analysis.GiaShowAsTypes.SpotDiagram;
    GeoImaSettings.Field.SetFieldNumber(1);
    GeoImaSettings.Wavelength.UseAllWavelengths();
    %GeoImaSettings.Surface.SetSurfaceNumber(54);
    GeoImaSettings.Surface.UseImageSurface();

    % change IMA file
    GeoImaSettings.File = append('dmd_', int2str(1000-1-i), '_MOS.IMA');

    % apply the new settings
    GeoIma.ApplyAndWaitForCompletion(); %be sure to have this line

    % get results
    GeoImaResults = GeoIma.GetResults();

    % save results to text file in Zemax\Samples\API\Matlab
    %GeoImaResults.GetTextFile(System.String.Concat(TheApplication.SamplesDir, '\test.txt'));
    filename = append('\data_', int2str(1000-1-i), '.txt');
    directory = 'C:\Users\Matthew\Desktop\Research\Fake_Data\new_fit\data';
    GeoImaResults.GetTextFile(System.String.Concat(directory, filename));
end

r = [];

end

function app = InitConnection()

import System.Reflection.*;

% Find the installed version of OpticStudio.
zemaxData = winqueryreg('HKEY_CURRENT_USER', 'Software\Zemax', 'ZemaxRoot');
NetHelper = strcat(zemaxData, '\ZOS-API\Libraries\ZOSAPI_NetHelper.dll');
% Note -- uncomment the following line to use a custom NetHelper path
% NetHelper = 'C:\Users\Matthew.Leung\Documents\Zemax\ZOS-API\Libraries\ZOSAPI_NetHelper.dll';
% This is the path to OpticStudio
NET.addAssembly(NetHelper);

success = ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize();
% Note -- uncomment the following line to use a custom initialization path
% success = ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize('C:\Program Files\OpticStudio\');
if success == 1
    LogMessage(strcat('Found OpticStudio at: ', char(ZOSAPI_NetHelper.ZOSAPI_Initializer.GetZemaxDirectory())));
else
    app = [];
    return;
end

% Now load the ZOS-API assemblies
NET.addAssembly(AssemblyName('ZOSAPI_Interfaces'));
NET.addAssembly(AssemblyName('ZOSAPI'));

% Create the initial connection class
TheConnection = ZOSAPI.ZOSAPI_Connection();

% Attempt to create a Standalone connection

% NOTE - if this fails with a message like 'Unable to load one or more of
% the requested types', it is usually caused by try to connect to a 32-bit
% version of OpticStudio from a 64-bit version of MATLAB (or vice-versa).
% This is an issue with how MATLAB interfaces with .NET, and the only
% current workaround is to use 32- or 64-bit versions of both applications.
app = TheConnection.CreateNewApplication();
if isempty(app)
   HandleError('An unknown connection error occurred!');
end
if ~app.IsValidLicenseForAPI
    HandleError('License check failed!');
    app = [];
end

end

function LogMessage(msg)
disp(msg);
end

function HandleError(error)
ME = MException('zosapi:HandleError', error);
throw(ME);
end

function  CleanupConnection(TheApplication)
% Note - this will close down the connection.

% If you want to keep the application open, you should skip this step
% and store the instance somewhere instead.
TheApplication.CloseApplication();
end

